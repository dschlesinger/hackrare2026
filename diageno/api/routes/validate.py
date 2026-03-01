"""/validate_schema endpoint — validate JSON against case Pydantic schema."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import ValidationError

from diageno.api.schemas import CaseInput, ValidateSchemaRequest, ValidationResponse

router = APIRouter()


@router.post("/validate_schema", response_model=ValidationResponse)
def validate_schema(req: ValidateSchemaRequest) -> ValidationResponse:
    """Validate a JSON payload against the CaseInput schema."""
    try:
        CaseInput(**req.data)
        return ValidationResponse(valid=True, errors=[])
    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        return ValidationResponse(valid=False, errors=errors)
    except Exception as e:
        return ValidationResponse(valid=False, errors=[str(e)])
