from typing import Any, Dict, List, Optional, AsyncIterator
import os
import httpx
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Thread
import webbrowser
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv

"""
Superset MCP Integration

This module provides a Model Control Protocol (MCP) server for Apache Superset,
enabling AI assistants to interact with and control a Superset instance programmatically.

It includes tools for:
- Authentication and token management
- Dashboard operations (list, get, create, update, delete)
- Chart management (list, get, create, update, delete)
- Database and dataset operations
- SQL execution and query management
- User information and recent activity tracking
- Advanced data type handling
- Tag management

Each tool follows a consistent naming convention: superset_<category>_<action>
"""

# Load environment variables from .env file
load_dotenv()

# Constants
SUPERSET_BASE_URL = os.getenv("SUPERSET_BASE_URL", "http://localhost:8088")
SUPERSET_USERNAME = os.getenv("SUPERSET_USERNAME")
SUPERSET_PASSWORD = os.getenv("SUPERSET_PASSWORD")
ACCESS_TOKEN_STORE_PATH = os.path.join(os.path.dirname(__file__), ".superset_token")

# Initialize FastAPI app for handling additional web endpoints if needed
app = FastAPI(title="Superset MCP Server")


@dataclass
class SupersetContext:
    """Typed context for the Superset MCP server"""

    client: httpx.AsyncClient
    base_url: str
    access_token: Optional[str] = None
    csrf_token: Optional[str] = None
    app: FastAPI = None


def load_stored_token() -> Optional[str]:
    """Load stored access token if it exists"""
    try:
        if os.path.exists(ACCESS_TOKEN_STORE_PATH):
            with open(ACCESS_TOKEN_STORE_PATH, "r") as f:
                return f.read().strip()
    except Exception:
        return None
    return None


def save_access_token(token: str):
    """Save access token to file"""
    try:
        with open(ACCESS_TOKEN_STORE_PATH, "w") as f:
            f.write(token)
    except Exception as e:
        print(f"Warning: Could not save access token: {e}")


@asynccontextmanager
async def superset_lifespan(server: FastMCP) -> AsyncIterator[SupersetContext]:
    """Manage application lifecycle for Superset integration"""
    print("Initializing Superset context...")

    # Create HTTP client
    client = httpx.AsyncClient(base_url=SUPERSET_BASE_URL, timeout=30.0)

    # Create context
    ctx = SupersetContext(client=client, base_url=SUPERSET_BASE_URL, app=app)

    # Try to load existing token
    stored_token = load_stored_token()
    if stored_token:
        ctx.access_token = stored_token
        # Set the token in the client headers
        client.headers.update({"Authorization": f"Bearer {stored_token}"})
        print("Using stored access token")

        # Verify token validity
        try:
            response = await client.get("/api/v1/me/")
            if response.status_code != 200:
                print(
                    f"Stored token is invalid (status {response.status_code}). Will need to re-authenticate."
                )
                ctx.access_token = None
                client.headers.pop("Authorization", None)
        except Exception as e:
            print(f"Error verifying stored token: {e}")
            ctx.access_token = None
            client.headers.pop("Authorization", None)

    try:
        yield ctx
    finally:
        # Cleanup on shutdown
        print("Shutting down Superset context...")
        await client.aclose()


# Initialize FastMCP server with lifespan and dependencies
mcp = FastMCP(
    "superset",
    lifespan=superset_lifespan,
    dependencies=["fastapi", "uvicorn", "python-dotenv", "httpx"],
)


@mcp.tool()
async def superset_auth_check_token_validity(ctx: Context) -> Dict[str, Any]:
    """
    Check if the current access token is still valid

    Returns status of token validity check
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"valid": False, "error": "No access token available"}

    try:
        # Make a simple API call to test if token is valid (get user info)
        response = await superset_ctx.client.get("/api/v1/me/")

        if response.status_code == 200:
            return {"valid": True}
        else:
            return {
                "valid": False,
                "status_code": response.status_code,
                "error": response.text,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


@mcp.tool()
async def superset_auth_refresh_token(ctx: Context) -> Dict[str, Any]:
    """
    Refresh the access token using the refresh endpoint

    Returns the new access token if successful
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "No access token to refresh. Please authenticate first."}

    try:
        # Use the refresh endpoint to get a new token
        response = await superset_ctx.client.post("/api/v1/security/refresh")

        if response.status_code != 200:
            return {
                "error": f"Failed to refresh token: {response.status_code} - {response.text}"
            }

        data = response.json()
        access_token = data.get("access_token")

        if not access_token:
            return {"error": "No access token returned from refresh"}

        # Save and set the new access token
        save_access_token(access_token)
        superset_ctx.access_token = access_token
        superset_ctx.client.headers.update({"Authorization": f"Bearer {access_token}"})

        return {
            "message": "Successfully refreshed access token",
            "access_token": access_token,
        }
    except Exception as e:
        return {"error": f"Error refreshing token: {str(e)}"}


@mcp.tool()
async def superset_auth_authenticate_user(
    ctx: Context,
    username: Optional[str] = None,
    password: Optional[str] = None,
    refresh: bool = True,
) -> Dict[str, Any]:
    """
    Authenticate with Superset and get access token

    If there's an existing token, will first try to check its validity.
    If invalid, will attempt to refresh token before falling back to re-authentication.

    Args:
        username: Superset username (or use env var)
        password: Superset password (or use env var)
        refresh: Whether to refresh the token if invalid (defaults to True)
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    # If we already have a token, check if it's valid
    if superset_ctx.access_token:
        validity = await superset_auth_check_token_validity(ctx)

        if validity.get("valid"):
            return {
                "message": "Already authenticated with valid token",
                "access_token": superset_ctx.access_token,
            }

        # Token invalid, try to refresh if requested
        if refresh:
            refresh_result = await superset_auth_refresh_token(ctx)
            if not refresh_result.get("error"):
                return refresh_result
            # If refresh fails, fall back to re-authentication

    # Use provided credentials or fall back to env vars
    username = username or SUPERSET_USERNAME
    password = password or SUPERSET_PASSWORD

    if not username or not password:
        return {
            "error": "Username and password must be provided either as arguments or set in environment variables"
        }

    try:
        # Get access token directly using the security login API endpoint
        response = await superset_ctx.client.post(
            "/api/v1/security/login",
            json={
                "username": username,
                "password": password,
                "provider": "db",
                "refresh": refresh,
            },
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to get access token: {response.status_code} - {response.text}"
            }

        data = response.json()
        access_token = data.get("access_token")

        if not access_token:
            return {"error": "No access token returned"}

        # Save and set the access token
        save_access_token(access_token)
        superset_ctx.access_token = access_token
        superset_ctx.client.headers.update({"Authorization": f"Bearer {access_token}"})

        return {
            "message": "Successfully authenticated with Superset",
            "access_token": access_token,
        }

    except Exception as e:
        return {"error": f"Authentication error: {str(e)}"}


async def with_auto_refresh(ctx: Context, api_call: callable) -> httpx.Response:
    """
    Helper function to handle automatic token refreshing for API calls

    This function will attempt to execute the provided API call. If the call
    fails with a 401 Unauthorized error, it will try to refresh the token
    and retry the API call once.

    Args:
        ctx: The MCP context
        api_call: The API call function to execute (should be a callable that returns a response)
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # First attempt
    try:
        response = await api_call()

        # If not an auth error, return the response
        if response.status_code != 401:
            return response

    except httpx.HTTPStatusError as e:
        if e.response.status_code != 401:
            raise e
        response = e.response
    except Exception as e:
        # For other errors, just raise
        raise e

    # If we got a 401, try to refresh the token
    print("Received 401 Unauthorized. Attempting to refresh token...")
    refresh_result = await superset_auth_refresh_token(ctx)

    if refresh_result.get("error"):
        # If refresh failed, try to re-authenticate
        print(
            f"Token refresh failed: {refresh_result.get('error')}. Attempting re-authentication..."
        )
        auth_result = await superset_auth_authenticate_user(ctx)

        if auth_result.get("error"):
            # If re-authentication failed, raise an exception
            raise HTTPException(status_code=401, detail="Authentication failed")

    # Retry the API call with the new token
    return await api_call()


@mcp.tool()
async def superset_dashboard_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of dashboards from Superset"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        # Use the auto-refresh helper for the API call
        async def api_call():
            return await superset_ctx.client.get("/api/v1/dashboard/")

        response = await with_auto_refresh(ctx, api_call)

        if response.status_code != 200:
            return {
                "error": f"Failed to get dashboards: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting dashboards: {str(e)}"}


@mcp.tool()
async def superset_dashboard_get_by_id(
    ctx: Context, dashboard_id: int
) -> Dict[str, Any]:
    """
    Get details for a specific dashboard

    Args:
        dashboard_id: ID of the dashboard to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/dashboard/{dashboard_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get dashboard: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting dashboard: {str(e)}"}


@mcp.tool()
async def superset_chart_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of charts from Superset"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/chart/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get charts: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting charts: {str(e)}"}


@mcp.tool()
async def superset_chart_get_by_id(ctx: Context, chart_id: int) -> Dict[str, Any]:
    """
    Get details for a specific chart

    Args:
        chart_id: ID of the chart to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/chart/{chart_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get chart: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting chart: {str(e)}"}


@mcp.tool()
async def superset_database_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of databases from Superset"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/database/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get databases: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting databases: {str(e)}"}


@mcp.tool()
async def superset_dataset_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of datasets from Superset"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/dataset/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get datasets: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting datasets: {str(e)}"}


@mcp.tool()
async def superset_sqllab_execute_query(
    ctx: Context, database_id: int, sql: str
) -> Dict[str, Any]:
    """
    Execute a SQL query in SQL Lab

    Args:
        database_id: ID of the database to query
        sql: SQL query to execute
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "database_id": database_id,
            "sql": sql,
            "client_id": f"mcp-client-{hash(sql)}",
            "schema": "",
            "sql_editor_id": f"mcp-editor-{hash(sql)}",
            "tab": "MCP Query",
            "async": False,
            "runAsync": False,
            "select_as_cta": False,
            "ctas_method": "TABLE",
        }

        response = await superset_ctx.client.post(
            "/api/v1/sqllab/execute/", json=payload
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to execute SQL: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error executing SQL: {str(e)}"}


@mcp.tool()
async def superset_activity_get_recent(ctx: Context) -> Dict[str, Any]:
    """Get recent activity data for the current user"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/log/recent_activity/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get recent activity: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting recent activity: {str(e)}"}


@mcp.tool()
async def superset_user_get_current(ctx: Context) -> Dict[str, Any]:
    """Get information about the currently authenticated user"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/me/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get user info: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting user info: {str(e)}"}


@mcp.tool()
async def superset_sqllab_get_saved_queries(ctx: Context) -> Dict[str, Any]:
    """Get a list of saved queries from SQL Lab"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/saved_query/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get saved queries: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting saved queries: {str(e)}"}


@mcp.tool()
async def superset_chart_create(
    ctx: Context,
    slice_name: str,
    datasource_id: int,
    datasource_type: str,
    viz_type: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a new chart in Superset

    Args:
        slice_name: Name of the chart
        datasource_id: ID of the dataset
        datasource_type: Type of datasource (table or query)
        viz_type: Visualization type (e.g., bar, line, etc.)
        params: Visualization parameters
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "slice_name": slice_name,
            "datasource_id": datasource_id,
            "datasource_type": datasource_type,
            "viz_type": viz_type,
            "params": params,
        }

        response = await superset_ctx.client.post("/api/v1/chart/", json=payload)

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create chart: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating chart: {str(e)}"}


@mcp.tool()
async def superset_database_get_tables(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get tables for a specific database

    Args:
        database_id: ID of the database
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(
            f"/api/v1/database/{database_id}/tables/"
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to get database tables: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting database tables: {str(e)}"}


@mcp.tool()
async def superset_dashboard_create(
    ctx: Context, dashboard_title: str, json_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a new dashboard in Superset

    Args:
        dashboard_title: Title of the dashboard
        json_metadata: Optional JSON metadata for dashboard configuration
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "dashboard_title": dashboard_title,
        }
        if json_metadata:
            payload["json_metadata"] = json_metadata

        response = await superset_ctx.client.post("/api/v1/dashboard/", json=payload)

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create dashboard: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating dashboard: {str(e)}"}


@mcp.tool()
async def superset_dashboard_update(
    ctx: Context, dashboard_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing dashboard

    Args:
        dashboard_id: ID of the dashboard to update
        data: Data to update (dictionary with fields to update)
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.put(
            f"/api/v1/dashboard/{dashboard_id}", json=data
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to update dashboard: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error updating dashboard: {str(e)}"}


@mcp.tool()
async def superset_dashboard_delete(ctx: Context, dashboard_id: int) -> Dict[str, Any]:
    """
    Delete a dashboard

    Args:
        dashboard_id: ID of the dashboard to delete
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.delete(f"/api/v1/dashboard/{dashboard_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to delete dashboard: {response.status_code} - {response.text}"
            }

        return {"message": f"Dashboard {dashboard_id} deleted successfully"}

    except Exception as e:
        return {"error": f"Error deleting dashboard: {str(e)}"}


@mcp.tool()
async def superset_chart_update(
    ctx: Context, chart_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing chart

    Args:
        chart_id: ID of the chart to update
        data: Data to update (dictionary with fields to update)
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.put(f"/api/v1/chart/{chart_id}", json=data)

        if response.status_code != 200:
            return {
                "error": f"Failed to update chart: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error updating chart: {str(e)}"}


@mcp.tool()
async def superset_chart_delete(ctx: Context, chart_id: int) -> Dict[str, Any]:
    """
    Delete a chart

    Args:
        chart_id: ID of the chart to delete
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.delete(f"/api/v1/chart/{chart_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to delete chart: {response.status_code} - {response.text}"
            }

        return {"message": f"Chart {chart_id} deleted successfully"}

    except Exception as e:
        return {"error": f"Error deleting chart: {str(e)}"}


@mcp.tool()
async def superset_dataset_get_by_id(ctx: Context, dataset_id: int) -> Dict[str, Any]:
    """
    Get details for a specific dataset

    Args:
        dataset_id: ID of the dataset to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/dataset/{dataset_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get dataset: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting dataset: {str(e)}"}


@mcp.tool()
async def superset_dataset_create(
    ctx: Context,
    table_name: str,
    database_id: int,
    schema: str = None,
    owners: List[int] = None,
) -> Dict[str, Any]:
    """
    Create a new dataset in Superset

    Args:
        table_name: Name of the table
        database_id: ID of the database
        schema: Optional schema name
        owners: Optional list of owner IDs
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "table_name": table_name,
            "database": database_id,
        }

        if schema:
            payload["schema"] = schema

        if owners:
            payload["owners"] = owners

        response = await superset_ctx.client.post("/api/v1/dataset/", json=payload)

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create dataset: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating dataset: {str(e)}"}


@mcp.tool()
async def superset_database_get_by_id(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Get details for a specific database

    Args:
        database_id: ID of the database to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/database/{database_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get database: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting database: {str(e)}"}


@mcp.tool()
async def superset_database_create(
    ctx: Context, database_name: str, sqlalchemy_uri: str, extra: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a new database connection in Superset

    Args:
        database_name: Name for the database connection
        sqlalchemy_uri: SQLAlchemy URI for the connection
        extra: Optional extra configuration parameters
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "database_name": database_name,
            "sqlalchemy_uri": sqlalchemy_uri,
        }

        if extra:
            payload["extra"] = extra

        response = await superset_ctx.client.post("/api/v1/database/", json=payload)

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create database: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating database: {str(e)}"}


@mcp.tool()
async def superset_query_stop(ctx: Context, client_id: str) -> Dict[str, Any]:
    """
    Stop a running query

    Args:
        client_id: Client ID of the query to stop
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "client_id": client_id,
        }

        response = await superset_ctx.client.post("/api/v1/query/stop", json=payload)

        if response.status_code != 200:
            return {
                "error": f"Failed to stop query: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error stopping query: {str(e)}"}


@mcp.tool()
async def superset_query_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of queries from Superset"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/query/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get queries: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting queries: {str(e)}"}


@mcp.tool()
async def superset_query_get_by_id(ctx: Context, query_id: int) -> Dict[str, Any]:
    """
    Get details for a specific query

    Args:
        query_id: ID of the query to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/query/{query_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get query: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting query: {str(e)}"}


@mcp.tool()
async def superset_tag_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of tags from Superset"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/tag/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get tags: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting tags: {str(e)}"}


@mcp.tool()
async def superset_tag_create(ctx: Context, name: str) -> Dict[str, Any]:
    """
    Create a new tag in Superset

    Args:
        name: Name for the tag
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "name": name,
        }

        response = await superset_ctx.client.post("/api/v1/tag/", json=payload)

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create tag: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating tag: {str(e)}"}


@mcp.tool()
async def superset_tag_get_by_id(ctx: Context, tag_id: int) -> Dict[str, Any]:
    """
    Get details for a specific tag

    Args:
        tag_id: ID of the tag to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/tag/{tag_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get tag: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting tag: {str(e)}"}


@mcp.tool()
async def superset_tag_objects(ctx: Context) -> Dict[str, Any]:
    """Get objects associated with tags"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/tag/get_objects/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get tagged objects: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting tagged objects: {str(e)}"}


@mcp.tool()
async def superset_explore_form_data_create(
    ctx: Context, form_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create form data for chart exploration

    Args:
        form_data: Form data for chart configuration
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.post(
            "/api/v1/explore/form_data", json=form_data
        )

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create form data: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating form data: {str(e)}"}


@mcp.tool()
async def superset_explore_form_data_get(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get form data for chart exploration

    Args:
        key: Key of the form data to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/explore/form_data/{key}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get form data: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting form data: {str(e)}"}


@mcp.tool()
async def superset_explore_permalink_create(
    ctx: Context, state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a permalink for chart exploration

    Args:
        state: State data for the permalink
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.post(
            "/api/v1/explore/permalink", json=state
        )

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create permalink: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating permalink: {str(e)}"}


@mcp.tool()
async def superset_explore_permalink_get(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get a permalink for chart exploration

    Args:
        key: Key of the permalink to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/explore/permalink/{key}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get permalink: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting permalink: {str(e)}"}


@mcp.tool()
async def superset_sqllab_format_sql(ctx: Context, sql: str) -> Dict[str, Any]:
    """
    Format a SQL query

    Args:
        sql: SQL query to format
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "sql": sql,
        }

        response = await superset_ctx.client.post(
            "/api/v1/sqllab/format_sql", json=payload
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to format SQL: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error formatting SQL: {str(e)}"}


@mcp.tool()
async def superset_sqllab_get_results(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get results of a SQL query

    Args:
        key: Result key to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/sqllab/results/?key={key}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get SQL results: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting SQL results: {str(e)}"}


@mcp.tool()
async def superset_sqllab_estimate_query_cost(
    ctx: Context, database_id: int, sql: str, schema: str = None
) -> Dict[str, Any]:
    """
    Estimate the cost of executing a SQL query

    Args:
        database_id: ID of the database
        sql: SQL query to estimate
        schema: Optional schema name
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "database_id": database_id,
            "sql": sql,
        }

        if schema:
            payload["schema"] = schema

        response = await superset_ctx.client.post(
            "/api/v1/sqllab/estimate", json=payload
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to estimate query cost: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error estimating query cost: {str(e)}"}


@mcp.tool()
async def superset_sqllab_export_query_results(
    ctx: Context, client_id: str
) -> Dict[str, Any]:
    """
    Export the results of a SQL query to CSV

    Args:
        client_id: Client ID of the query
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/sqllab/export/{client_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to export query results: {response.status_code} - {response.text}"
            }

        return {"message": "Query results exported successfully", "data": response.text}

    except Exception as e:
        return {"error": f"Error exporting query results: {str(e)}"}


@mcp.tool()
async def superset_sqllab_get_bootstrap_data(ctx: Context) -> Dict[str, Any]:
    """Get the bootstrap data for SQL Lab"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/sqllab/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get SQL Lab bootstrap data: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting SQL Lab bootstrap data: {str(e)}"}


@mcp.tool()
async def superset_user_get_roles(ctx: Context) -> Dict[str, Any]:
    """Get roles for the current user"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/me/roles/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get user roles: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting user roles: {str(e)}"}


@mcp.tool()
async def superset_menu_get(ctx: Context) -> Dict[str, Any]:
    """Get the Superset menu data"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/menu/")

        if response.status_code != 200:
            return {
                "error": f"Failed to get menu data: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting menu data: {str(e)}"}


@mcp.tool()
async def superset_advanced_data_type_convert(
    ctx: Context, type_name: str, value: Any
) -> Dict[str, Any]:
    """
    Convert a value to an advanced data type

    Args:
        type_name: Name of the advanced data type
        value: Value to convert
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        payload = {
            "type_name": type_name,
            "value": value,
        }

        response = await superset_ctx.client.get(
            "/api/v1/advanced_data_type/convert", params=payload
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to convert advanced data type: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error converting advanced data type: {str(e)}"}


@mcp.tool()
async def superset_advanced_data_type_list(ctx: Context) -> Dict[str, Any]:
    """Get list of available advanced data types"""
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get("/api/v1/advanced_data_type/types")

        if response.status_code != 200:
            return {
                "error": f"Failed to get advanced data types: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting advanced data types: {str(e)}"}


@mcp.tool()
async def superset_database_schemas(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Get schemas for a specific database

    Args:
        database_id: ID of the database
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(
            f"/api/v1/database/{database_id}/schemas/"
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to get database schemas: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting database schemas: {str(e)}"}


@mcp.tool()
async def superset_database_test_connection(
    ctx: Context, database_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Test a database connection

    Args:
        database_data: Database connection data
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.post(
            "/api/v1/database/test_connection", json=database_data
        )

        if response.status_code != 200:
            return {
                "error": f"Failed to test connection: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error testing connection: {str(e)}"}


@mcp.tool()
async def superset_saved_query_get_by_id(ctx: Context, query_id: int) -> Dict[str, Any]:
    """
    Get details for a specific saved query

    Args:
        query_id: ID of the saved query to retrieve
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.get(f"/api/v1/saved_query/{query_id}")

        if response.status_code != 200:
            return {
                "error": f"Failed to get saved query: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error getting saved query: {str(e)}"}


@mcp.tool()
async def superset_saved_query_create(
    ctx: Context, query_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new saved query

    Args:
        query_data: Saved query data including db_id, schema, sql, label, etc.
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

    if not superset_ctx.access_token:
        return {"error": "Not authenticated. Please authenticate first."}

    try:
        response = await superset_ctx.client.post(
            "/api/v1/saved_query/", json=query_data
        )

        if response.status_code not in [200, 201]:
            return {
                "error": f"Failed to create saved query: {response.status_code} - {response.text}"
            }

        return response.json()

    except Exception as e:
        return {"error": f"Error creating saved query: {str(e)}"}


if __name__ == "__main__":
    print("Starting Superset MCP server...")
    mcp.run()
