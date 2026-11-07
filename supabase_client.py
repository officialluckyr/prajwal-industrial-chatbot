
import os
from supabase import create_client


def get_supabase_client():
    """
    Create and return a Supabase client.
    Credentials are loaded from environment variables.
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY")

    if not supabase_url:
        raise ValueError("SUPABASE_URL is missing.")

    if not supabase_key:
        raise ValueError("SUPABASE_SECRET_KEY is missing.")

    return create_client(supabase_url, supabase_key)
