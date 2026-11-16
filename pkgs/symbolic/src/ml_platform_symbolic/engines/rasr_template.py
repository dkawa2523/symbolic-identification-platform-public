"""Explicit rasr proposals plus common joint template fitting, not native conditioning."""
from . import pretrained_template


def capabilities():
    return pretrained_template.capabilities("rasr")

def validate_request(request, context):
    return pretrained_template.validate_request(request, context, engine="rasr")

def discover(request, context):
    return pretrained_template.discover(request, context, engine="rasr")
