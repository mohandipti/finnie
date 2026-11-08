"""CDK app entry point. Run CDK commands from the infra/ folder: `cdk synth`, `cdk diff`, `cdk deploy`."""

import os

import aws_cdk as cdk
from finnie_stack import FinnieStack

app = cdk.App()
# The lab account is shared, so each person deploys under their own stack name (cdk.json or
# `-c stackName=...`); two deploys with the same name would replace each other's stack.
stack_name = app.node.try_get_context("stackName") or "FinnieStack"
FinnieStack(
    app,
    stack_name,
    # The lab account only allows us-east-1; override with `-c region=...` in other accounts.
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    # Deploy and publish assets with your own CLI credentials. The lab account can't assume the
    # roles that the default synthesizer uses (sts:AssumeRole is denied).
    synthesizer=cdk.CliCredentialsStackSynthesizer(),
    description="Finnie AI finance assistant: Streamlit in Docker on EC2",
)
app.synth()
