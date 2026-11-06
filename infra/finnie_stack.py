"""AWS infrastructure for Finnie: the Docker image on a single EC2 instance.

    Your IP ─► HTTP :80 ─► EC2 t2.micro (Amazon Linux 2023, default VPC)
                              └─ Docker: Finnie image from ECR (Streamlit :8501)
                                   ├─ API keys and MongoDB URI from Secrets Manager (fetched at boot)
                                   ├─ logs to CloudWatch
                                   └─ MongoDB Atlas (outside AWS; allows only the Elastic IP)

Designed for the course lab account, which allows EC2 t2.micro in us-east-1 but not ECS or load
balancers. A new image (any code or index change) replaces the instance on the next deploy.

Settings come from CDK context (cdk.json, or `cdk deploy -c key=value`):
    secretName              secret holding the Google API key (default: finnie/google-api-key)
    alphaVantageSecretName  secret holding the Alpha Vantage key (default: finnie/alpha-vantage-api-key)
    serpApiSecretName       secret holding the SerpAPI key (default: finnie/serp-api-key)
    mongodbSecretName       secret holding the MongoDB connection string (default: finnie/mongodb-uri)
    allowedCidrs            comma-separated IP ranges allowed to reach the app (default: everyone; set this!)
    instanceType            EC2 instance type (default: t2.micro)
"""

from pathlib import Path

from aws_cdk import Annotations, CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from constructs import Construct

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOST_PORT = 80
CONTAINER_PORT = 8501

# Environment variable → (context setting, default secret name, required). The app runs without the
# optional keys (no live prices or news), so a missing optional secret doesn't stop the instance starting.
SECRETS = {
    "GOOGLE_API_KEY": ("secretName", "finnie/google-api-key", True),
    "ALPHA_VANTAGE_API_KEY": ("alphaVantageSecretName", "finnie/alpha-vantage-api-key", False),
    "SERP_API_KEY": ("serpApiSecretName", "finnie/serp-api-key", False),
    "MONGODB_URI": ("mongodbSecretName", "finnie/mongodb-uri", False),
}

# Starts the container after Docker on every boot, fetching the secrets afresh each time.
FINNIE_SERVICE = """[Unit]
Description=Finnie container
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/finnie/start.sh

[Install]
WantedBy=multi-user.target"""


class FinnieStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        secret_names = {env: self.node.try_get_context(setting) or default
                        for env, (setting, default, _) in SECRETS.items()}
        allowed_cidrs = [c.strip() for c in (self.node.try_get_context("allowedCidrs") or "").split(",") if c.strip()]
        instance_type = self.node.try_get_context("instanceType") or "t2.micro"

        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # t2.micro is x86, so the image is built for amd64 (Docker emulates it on Apple Silicon).
        image = ecr_assets.DockerImageAsset(
            self, "Image",
            directory=str(PROJECT_ROOT),
            platform=ecr_assets.Platform.LINUX_AMD64,
            exclude=["infra/cdk.out", ".venv*", ".git"],
        )
        secrets = [secretsmanager.Secret.from_secret_name_v2(self, f"Secret{env}", name)
                   for env, name in secret_names.items()]
        log_group = logs.LogGroup(
            self, "Logs", retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.DESTROY
        )

        security_group = ec2.SecurityGroup(
            self, "WebSecurityGroup", vpc=vpc, description="Finnie web access", allow_all_outbound=True
        )
        if allowed_cidrs:
            for cidr in allowed_cidrs:
                security_group.add_ingress_rule(ec2.Peer.ipv4(cidr), ec2.Port.tcp(HOST_PORT), f"Finnie from {cidr}")
        else:
            security_group.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(HOST_PORT), "Finnie from anywhere")
            Annotations.of(self).add_warning(
                "Finnie is open to the internet and uses your paid API key; deploy with -c allowedCidrs=<your-ip>/32"
            )

        role = iam.Role(
            self, "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
        )
        image.repository.grant_pull(role)
        for secret in secrets:
            secret.grant_read(role)
        log_group.grant_write(role)

        region, account = self.region, self.account
        registry = f"{account}.dkr.ecr.{region}.amazonaws.com"

        # Runs on every boot (via finnie.service), so a changed secret takes effect after a reboot.
        start_lines = [
            "#!/bin/bash",
            "set -euo pipefail",  # no `set -x`: it would print the API key to the console log
            f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry}",
            "umask 077",
            "printf 'LOG_LEVEL=INFO\\n' > /opt/finnie/env.new",
        ]
        for env, name in secret_names.items():
            fetch = (f"KEY=$(aws secretsmanager get-secret-value --region {region} --secret-id {name} "
                     "--query SecretString --output text)")
            start_lines += [
                fetch if SECRETS[env][2] else fetch + " || KEY=''",
                f"if [ -n \"$KEY\" ]; then printf '{env}=%s\\n' \"$KEY\" >> /opt/finnie/env.new; fi; unset KEY",
            ]
        start_lines += [
            "mv /opt/finnie/env.new /opt/finnie/env",
            "docker rm -f finnie 2>/dev/null || true",
            (f"docker run -d --name finnie --restart always -p {HOST_PORT}:{CONTAINER_PORT} "
            "--env-file /opt/finnie/env "
            f"--log-driver awslogs --log-opt awslogs-region={region} --log-opt awslogs-group={log_group.log_group_name} "
            f"{image.image_uri}"),
        ]
        start_script = "\n".join(start_lines)

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -euo pipefail",  # no `set -x`: it would print the API key to the console log
            "dnf install -y docker",
            "systemctl enable --now docker",
            # 1 GB of swap as headroom on a 1 GB instance
            "fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile",
            "echo '/swapfile none swap sw 0 0' >> /etc/fstab",
            "install -d -m 700 /opt/finnie",
            f"cat > /opt/finnie/start.sh <<'EOF'\n{start_script}\nEOF",
            "chmod 700 /opt/finnie/start.sh",
            f"cat > /etc/systemd/system/finnie.service <<'EOF'\n{FINNIE_SERVICE}\nEOF",
            "systemctl daemon-reload",
            "systemctl enable --now finnie.service",
        )

        instance = ec2.Instance(
            self, "Instance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            instance_type=ec2.InstanceType(instance_type),
            machine_image=ec2.MachineImage.latest_amazon_linux2023(cpu_type=ec2.AmazonLinuxCpuType.X86_64),
            security_group=security_group,
            role=role,
            user_data=user_data,
            user_data_causes_replacement=True,  # a new image means a fresh instance running it
            require_imdsv2=True,
            associate_public_ip_address=True,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/xvda",
                volume=ec2.BlockDeviceVolume.ebs(16, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True),
            )],
        )

        # A fixed public IP that moves to each replacement instance, so MongoDB Atlas only needs to
        # allow this one address. It's released only by `cdk destroy`.
        elastic_ip = ec2.CfnEIP(self, "ElasticIp", domain="vpc", instance_id=instance.instance_id)

        CfnOutput(self, "AppUrl", value=f"http://{elastic_ip.ref}",
                  description="Open this URL to use Finnie (allow ~5 minutes after deploy for start-up)")
        CfnOutput(self, "ElasticIpAddress", value=elastic_ip.ref,
                  description="Add this address as <ip>/32 to MongoDB Atlas Network Access")
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name)
