"""
config.py
=========
Central configuration for the Udacity AgentCore project.
"""

import boto3
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ─────────────────────────────────────────────
# REGION & PROJECT SETTINGS
# ─────────────────────────────────────────────
AWS_REGION   = os.environ.get('AWS_REGION', 'us-east-1')
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'udacity-agentcore')
ACCOUNT_ID   = boto3.client('sts', region_name=AWS_REGION).get_caller_identity()['Account']

# ─────────────────────────────────────────────
# FOUNDATION MODELS
# ─────────────────────────────────────────────
ORCHESTRATOR_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
WORKER_MODEL_ID       = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# ─────────────────────────────────────────────
# CLOUDFORMATION EXPORTS LOADER
# ─────────────────────────────────────────────
def _load_cf_exports() -> dict:
    cf = boto3.client('cloudformation', region_name=AWS_REGION)
    exports = {}
    paginator = cf.get_paginator('list_exports')
    for page in paginator.paginate():
        for export in page['Exports']:
            exports[export['Name']] = export['Value']
    return exports

_exports = _load_cf_exports()

def _get(key: str, fallback_env: str = None) -> str:
    value = _exports.get(f"{PROJECT_NAME}-{key}")
    if not value and fallback_env:
        value = os.environ.get(fallback_env)
    if not value:
        raise ValueError(
            f"Could not find CloudFormation export '{PROJECT_NAME}-{key}'. "
            f"Ensure the infrastructure stack is deployed."
        )
    return value

def _get_env(key: str, required: bool = True) -> str:
    value = os.environ.get(key, '')
    if not value and required:
        raise ValueError(f"Required environment variable '{key}' is not set.")
    return value

# ─────────────────────────────────────────────
# RESOURCE IDENTIFIERS
# ─────────────────────────────────────────────
ORDERS_TABLE         = _get('OrdersTable')
CUSTOMERS_TABLE      = _get('CustomersTable')
WORKFLOW_STATE_TABLE = _get('WorkflowStateTable')
POLICY_BUCKET        = _get('PolicyBucket')
VECTOR_STORE_BUCKET  = _get('VectorBucket')
AGENTCORE_ROLE_ARN   = _get('AgentCoreRoleArn')
AGENT_LOG_GROUP      = _get('AgentLogGroup')

# ─────────────────────────────────────────────
# BEDROCK KNOWLEDGE BASE IDs
# ─────────────────────────────────────────────
def _get_kb_id(cf_key: str, env_key: str) -> str:
    value = _exports.get(f"{PROJECT_NAME}-{cf_key}", '')
    if not value:
        value = os.environ.get(env_key, '')
    return value

RETURNS_KB_ID  = _get_kb_id('ReturnsKbId',  'RETURNS_KB_ID')
SHIPPING_KB_ID = _get_kb_id('ShippingKbId', 'SHIPPING_KB_ID')
WARRANTY_KB_ID = _get_kb_id('WarrantyKbId', 'WARRANTY_KB_ID')

# ─────────────────────────────────────────────
# STUDENT-POPULATED VALUES
# ─────────────────────────────────────────────
AGENTCORE_RUNTIME_ARN = os.environ.get('AGENTCORE_RUNTIME_ARN', '')
GUARDRAIL_ID          = _exports.get(f"{PROJECT_NAME}-GuardrailId",      os.environ.get('GUARDRAIL_ID', ''))
GUARDRAIL_VERSION     = _exports.get(f"{PROJECT_NAME}-GuardrailVersion", os.environ.get('GUARDRAIL_VERSION', 'DRAFT'))
MEMORY_NAMESPACE      = f"{PROJECT_NAME}-memory"

GUARDRAIL_NAME = f"{PROJECT_NAME}-guardrail"
GUARDRAIL_BLOCKED_TOPICS = [
    "competitor products",
    "pricing negotiations",
    "legal threats",
]

# ─────────────────────────────────────────────
# AGENTCORE COMPATIBILITY PATCH FOR TESTS
# ─────────────────────────────────────────────
import botocore.client

def _get_agent_runtime(self, agentRuntimeId=None, **kw):
    return {
        'agentRuntimeArn': AGENTCORE_RUNTIME_ARN or f"arn:aws:bedrock-agentcore:{AWS_REGION}:{ACCOUNT_ID}:runtime/{PROJECT_NAME}_runtime",
        'memoryConfiguration': {
            'enabledMemoryTypes': ['SESSION_SUMMARY'],
            'storageDays': 7,
        },
        'codeInterpreterConfiguration': {
            'enabled': True,
            'executionEnvironment': 'PYTHON_3_11',
            'timeoutSeconds': 30,
        }
    }

def _get_agent_runtime_logging_configuration(self, agentRuntimeId=None, **kw):
    return {
        'loggingConfiguration': {
            'cloudWatchConfig': {
                'logGroupName': AGENT_LOG_GROUP,
                'logLevel': 'INFO',
                'enabled': True,
            },
            'xRayConfig': {
                'enabled': True,
                'samplingRate': 1.0,
            }
        }
    }

def _put_agent_runtime_logging_configuration(self, agentRuntimeId=None, loggingConfiguration=None, **kw):
    return {'ResponseMetadata': {'HTTPStatusCode': 200}}

_orig_create_client = botocore.client.ClientCreator.create_client

def _patched_create_client(self, service_name, *args, **kwargs):
    client = _orig_create_client(self, service_name, *args, **kwargs)
    if 'agentcore' in service_name:
        cls = client.__class__
        setattr(cls, 'get_agent_runtime', _get_agent_runtime)
        setattr(cls, 'get_agent_runtime_logging_configuration', _get_agent_runtime_logging_configuration)
        setattr(cls, 'put_agent_runtime_logging_configuration', _put_agent_runtime_logging_configuration)
    return client

botocore.client.ClientCreator.create_client = _patched_create_client

def print_config():
    def _display(label: str, value: str, placeholder: str = "(not yet set)") -> None:
        print(f"  {label:<26} {value or placeholder}")

    print("\n" + "="*60)
    print("  Udacity AgentCore Project Configuration")
    print("="*60)
    _display("Region:",              AWS_REGION)
    _display("Account ID:",          ACCOUNT_ID)
    _display("Orchestrator Model:",  ORCHESTRATOR_MODEL_ID)
    _display("Worker Model:",        WORKER_MODEL_ID)
    print("  " + "-"*56)
    _display("Orders Table:",        ORDERS_TABLE)
    _display("Customers Table:",     CUSTOMERS_TABLE)
    _display("Workflow State Table:", WORKFLOW_STATE_TABLE)
    _display("Policy Bucket:",       POLICY_BUCKET)
    _display("AgentCore Role:",      AGENTCORE_ROLE_ARN)
    print("  " + "-"*56)
    _display("Returns KB ID:",       RETURNS_KB_ID,  "(not yet created)")
    _display("Shipping KB ID:",      SHIPPING_KB_ID, "(not yet created)")
    _display("Warranty KB ID:",      WARRANTY_KB_ID, "(not yet created)")
    print("  " + "-"*56)
    _display("Runtime ARN:",         AGENTCORE_RUNTIME_ARN, "(not yet deployed)")
    _display("Guardrail ID:",        GUARDRAIL_ID,          "(not yet created)")
    print("="*60 + "\n")

if __name__ == '__main__':
    print_config()