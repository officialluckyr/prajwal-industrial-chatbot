
import os
import uuid

from supabase import create_client


def get_supabase_client():
    """Create a Supabase client using environment variables."""

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing.")

    if not supabase_key:
        raise ValueError("SUPABASE_SECRET_KEY is missing.")

    return create_client(supabase_url, supabase_key)


def validate_user_id(user_id):
    """Require a valid UUID for every user-scoped operation."""

    try:
        return str(uuid.UUID(str(user_id)))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("A valid user UUID is required.")


def add_equipment(user_id, name, **details):
    """Insert equipment and return the saved record."""

    user_id = validate_user_id(user_id)

    if not name or not name.strip():
        raise ValueError("Equipment name is required.")

    allowed_fields = {
        "equipment_type",
        "manufacturer",
        "model_number",
        "serial_number",
        "location",
        "status",
        "installation_date",
        "last_maintenance_at",
        "next_maintenance_at",
        "notes",
        "metadata",
    }

    record = {
        "user_id": user_id,
        "name": name.strip(),
    }

    for key, value in details.items():
        if key not in allowed_fields:
            raise ValueError(f"Unsupported equipment field: {key}")
        if value is not None:
            record[key] = value

    response = (
        get_supabase_client()
        .table("equipment")
        .insert(record)
        .execute()
    )

    if not response.data:
        raise RuntimeError("Equipment could not be saved.")

    return response.data[0]


def list_equipment(user_id):
    """List only equipment belonging to the specified user."""

    user_id = validate_user_id(user_id)

    response = (
        get_supabase_client()
        .table("equipment")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    return response.data or []


def get_equipment(user_id, equipment_id):
    """Retrieve one equipment record owned by the specified user."""

    user_id = validate_user_id(user_id)
    equipment_id = str(uuid.UUID(str(equipment_id)))

    response = (
        get_supabase_client()
        .table("equipment")
        .select("*")
        .eq("user_id", user_id)
        .eq("id", equipment_id)
        .limit(1)
        .execute()
    )

    return response.data[0] if response.data else None
