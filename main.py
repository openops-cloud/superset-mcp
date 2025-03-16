from typing import (
    Any,
    Dict,
    List,
    Optional,
    AsyncIterator,
    Callable,
    TypeVar,
    Awaitable,
)
import os
import httpx
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
import inspect
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

# Type variables for generic function annotations
T = TypeVar("T")
R = TypeVar("R")

# ===== Helper Functions and Decorators =====


def requires_auth(
    func: Callable[..., Awaitable[Dict[str, Any]]],
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Decorator to check authentication before executing a function"""

    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs) -> Dict[str, Any]:
        superset_ctx: SupersetContext = ctx.request_context.lifespan_context

        if not superset_ctx.access_token:
            return {"error": "Not authenticated. Please authenticate first."}

        return await func(ctx, *args, **kwargs)

    return wrapper


def handle_api_errors(
    func: Callable[..., Awaitable[Dict[str, Any]]],
) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """Decorator to handle API errors in a consistent way"""

    @wraps(func)
    async def wrapper(ctx: Context, *args, **kwargs) -> Dict[str, Any]:
        try:
            return await func(ctx, *args, **kwargs)
        except Exception as e:
            # Extract function name for better error context
            function_name = func.__name__
            return {"error": f"Error in {function_name}: {str(e)}"}

    return wrapper


async def with_auto_refresh(
    ctx: Context, api_call: Callable[[], Awaitable[httpx.Response]]
) -> httpx.Response:
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


async def get_csrf_token(ctx: Context) -> Optional[str]:
    """
    Get a CSRF token from Superset

    Makes a request to the /api/v1/security/csrf_token endpoint to get a token

    Args:
        ctx: MCP context
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    client = superset_ctx.client

    try:
        response = await client.get("/api/v1/security/csrf_token/")
        if response.status_code == 200:
            data = response.json()
            csrf_token = data.get("result")
            superset_ctx.csrf_token = csrf_token
            return csrf_token
        else:
            print(f"Failed to get CSRF token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error getting CSRF token: {str(e)}")
        return None


async def make_api_request(
    ctx: Context,
    method: str,
    endpoint: str,
    data: Dict[str, Any] = None,
    params: Dict[str, Any] = None,
    auto_refresh: bool = True,
) -> Dict[str, Any]:
    """
    Helper function to make API requests to Superset

    Args:
        ctx: MCP context
        method: HTTP method (get, post, put, delete)
        endpoint: API endpoint (without base URL)
        data: Optional JSON payload for POST/PUT requests
        params: Optional query parameters
        auto_refresh: Whether to auto-refresh token on 401
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    client = superset_ctx.client

    # For non-GET requests, make sure we have a CSRF token
    if method.lower() != "get" and not superset_ctx.csrf_token:
        await get_csrf_token(ctx)

    async def make_request() -> httpx.Response:
        headers = {}

        # Add CSRF token for non-GET requests
        if method.lower() != "get" and superset_ctx.csrf_token:
            headers["X-CSRFToken"] = superset_ctx.csrf_token

        if method.lower() == "get":
            return await client.get(endpoint, params=params)
        elif method.lower() == "post":
            return await client.post(
                endpoint, json=data, params=params, headers=headers
            )
        elif method.lower() == "put":
            return await client.put(endpoint, json=data, headers=headers)
        elif method.lower() == "delete":
            return await client.delete(endpoint, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

    # Use auto_refresh if requested
    response = (
        await with_auto_refresh(ctx, make_request)
        if auto_refresh
        else await make_request()
    )

    if response.status_code not in [200, 201]:
        return {
            "error": f"API request failed: {response.status_code} - {response.text}"
        }

    return response.json()


# ===== Authentication Tools =====


@mcp.tool()
@handle_api_errors
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
@handle_api_errors
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
@handle_api_errors
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

        # Get CSRF token after successful authentication
        await get_csrf_token(ctx)

        return {
            "message": "Successfully authenticated with Superset",
            "access_token": access_token,
        }

    except Exception as e:
        return {"error": f"Authentication error: {str(e)}"}


# ===== Dashboard Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of dashboards from Superset"""
    return await make_api_request(ctx, "get", "/api/v1/dashboard/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_get_by_id(
    ctx: Context, dashboard_id: int
) -> Dict[str, Any]:
    """
    Get details for a specific dashboard

    Args:
        dashboard_id: ID of the dashboard to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/dashboard/{dashboard_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_create(
    ctx: Context, dashboard_title: str, json_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a new dashboard in Superset

    Args:
        dashboard_title: Title of the dashboard
        json_metadata: Optional JSON metadata for dashboard configuration
    """
    payload = {"dashboard_title": dashboard_title}
    if json_metadata:
        payload["json_metadata"] = json_metadata

    return await make_api_request(ctx, "post", "/api/v1/dashboard/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_update(
    ctx: Context, dashboard_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing dashboard

    Args:
        dashboard_id: ID of the dashboard to update
        data: Data to update (dictionary with fields to update)
    """
    return await make_api_request(
        ctx, "put", f"/api/v1/dashboard/{dashboard_id}", data=data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dashboard_delete(ctx: Context, dashboard_id: int) -> Dict[str, Any]:
    """
    Delete a dashboard

    Args:
        dashboard_id: ID of the dashboard to delete
    """
    response = await make_api_request(
        ctx, "delete", f"/api/v1/dashboard/{dashboard_id}"
    )

    # For delete endpoints, we may want a custom success message
    if not response.get("error"):
        return {"message": f"Dashboard {dashboard_id} deleted successfully"}

    return response


# ===== Chart Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of charts from Superset"""
    return await make_api_request(ctx, "get", "/api/v1/chart/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_get_by_id(ctx: Context, chart_id: int) -> Dict[str, Any]:
    """
    Get details for a specific chart

    Args:
        chart_id: ID of the chart to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/chart/{chart_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
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
    payload = {
        "slice_name": slice_name,
        "datasource_id": datasource_id,
        "datasource_type": datasource_type,
        "viz_type": viz_type,
        "params": params,
    }

    return await make_api_request(ctx, "post", "/api/v1/chart/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_update(
    ctx: Context, chart_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing chart

    Args:
        chart_id: ID of the chart to update
        data: Data to update (dictionary with fields to update)
    """
    return await make_api_request(ctx, "put", f"/api/v1/chart/{chart_id}", data=data)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_chart_delete(ctx: Context, chart_id: int) -> Dict[str, Any]:
    """
    Delete a chart

    Args:
        chart_id: ID of the chart to delete
    """
    response = await make_api_request(ctx, "delete", f"/api/v1/chart/{chart_id}")

    if not response.get("error"):
        return {"message": f"Chart {chart_id} deleted successfully"}

    return response


# ===== Database Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of databases from Superset"""
    return await make_api_request(ctx, "get", "/api/v1/database/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_by_id(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Get details for a specific database

    Args:
        database_id: ID of the database to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/database/{database_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
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
    payload = {
        "database_name": database_name,
        "sqlalchemy_uri": sqlalchemy_uri,
    }

    if extra:
        payload["extra"] = extra

    return await make_api_request(ctx, "post", "/api/v1/database/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_tables(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get a list of tables for a given database

    Args:
        database_id: ID of the database
    """
    return await make_api_request(ctx, "get", f"/api/v1/database/{database_id}/tables/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_schemas(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Get schemas for a specific database

    Args:
        database_id: ID of the database
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/schemas/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_test_connection(
    ctx: Context, database_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Test a database connection

    Args:
        database_data: Database connection data
    """
    return await make_api_request(
        ctx, "post", "/api/v1/database/test_connection", data=database_data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_update(
    ctx: Context, database_id: int, data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an existing database

    Args:
        database_id: ID of the database to update
        data: Data to update (dictionary with fields to update)
    """
    return await make_api_request(
        ctx, "put", f"/api/v1/database/{database_id}", data=data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_delete(ctx: Context, database_id: int) -> Dict[str, Any]:
    """
    Delete a database

    Args:
        database_id: ID of the database to delete
    """
    response = await make_api_request(ctx, "delete", f"/api/v1/database/{database_id}")

    if not response.get("error"):
        return {"message": f"Database {database_id} deleted successfully"}

    return response


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_catalogs(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get all catalogs from a database

    Args:
        database_id: ID of the database
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/catalogs/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_connection(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get a database connection info

    Args:
        database_id: ID of the database
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/connection"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_function_names(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get function names supported by a database

    Args:
        database_id: ID of the database
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/function_names/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_get_related_objects(
    ctx: Context, database_id: int
) -> Dict[str, Any]:
    """
    Get charts and dashboards count associated to a database

    Args:
        database_id: ID of the database
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/database/{database_id}/related_objects/"
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_validate_sql(
    ctx: Context, database_id: int, sql: str
) -> Dict[str, Any]:
    """
    Validate arbitrary SQL

    Args:
        database_id: ID of the database
        sql: SQL to validate
    """
    payload = {"sql": sql}
    return await make_api_request(
        ctx, "post", f"/api/v1/database/{database_id}/validate_sql/", data=payload
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_database_validate_parameters(
    ctx: Context, parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate database connection parameters

    Args:
        parameters: Connection parameters to validate
    """
    return await make_api_request(
        ctx, "post", "/api/v1/database/validate_parameters/", data=parameters
    )


# ===== Dataset Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of datasets from Superset"""
    return await make_api_request(ctx, "get", "/api/v1/dataset/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_dataset_get_by_id(ctx: Context, dataset_id: int) -> Dict[str, Any]:
    """
    Get details for a specific dataset

    Args:
        dataset_id: ID of the dataset to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/dataset/{dataset_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
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
    payload = {
        "table_name": table_name,
        "database": database_id,
    }

    if schema:
        payload["schema"] = schema

    if owners:
        payload["owners"] = owners

    return await make_api_request(ctx, "post", "/api/v1/dataset/", data=payload)


# ===== SQL Lab Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_execute_query(
    ctx: Context, database_id: int, sql: str
) -> Dict[str, Any]:
    """
    Execute a SQL query in SQL Lab

    Args:
        database_id: ID of the database to query
        sql: SQL query to execute
    """
    # Ensure we have a CSRF token before executing the query
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context
    if not superset_ctx.csrf_token:
        await get_csrf_token(ctx)

    payload = {
        "database_id": database_id,
        "sql": sql,
        "schema": "",
        "tab": "MCP Query",
        "runAsync": False,
        "select_as_cta": False,
    }

    return await make_api_request(ctx, "post", "/api/v1/sqllab/execute/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_get_saved_queries(ctx: Context) -> Dict[str, Any]:
    """Get a list of saved queries from SQL Lab"""
    return await make_api_request(ctx, "get", "/api/v1/saved_query/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_format_sql(ctx: Context, sql: str) -> Dict[str, Any]:
    """
    Format a SQL query

    Args:
        sql: SQL query to format
    """
    payload = {"sql": sql}
    return await make_api_request(
        ctx, "post", "/api/v1/sqllab/format_sql", data=payload
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_get_results(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get results of a SQL query

    Args:
        key: Result key to retrieve
    """
    return await make_api_request(
        ctx, "get", f"/api/v1/sqllab/results/", params={"key": key}
    )


@mcp.tool()
@requires_auth
@handle_api_errors
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
    payload = {
        "database_id": database_id,
        "sql": sql,
    }

    if schema:
        payload["schema"] = schema

    return await make_api_request(ctx, "post", "/api/v1/sqllab/estimate", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_sqllab_export_query_results(
    ctx: Context, client_id: str
) -> Dict[str, Any]:
    """
    Export the results of a SQL query to CSV

    Args:
        client_id: Client ID of the query
    """
    superset_ctx: SupersetContext = ctx.request_context.lifespan_context

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
@requires_auth
@handle_api_errors
async def superset_sqllab_get_bootstrap_data(ctx: Context) -> Dict[str, Any]:
    """Get the bootstrap data for SQL Lab"""
    return await make_api_request(ctx, "get", "/api/v1/sqllab/")


# ===== Saved Query Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_saved_query_get_by_id(ctx: Context, query_id: int) -> Dict[str, Any]:
    """
    Get details for a specific saved query

    Args:
        query_id: ID of the saved query to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/saved_query/{query_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_saved_query_create(
    ctx: Context, query_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new saved query

    Args:
        query_data: Saved query data including db_id, schema, sql, label, etc.
    """
    return await make_api_request(ctx, "post", "/api/v1/saved_query/", data=query_data)


# ===== Query Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_query_stop(ctx: Context, client_id: str) -> Dict[str, Any]:
    """
    Stop a running query

    Args:
        client_id: Client ID of the query to stop
    """
    payload = {"client_id": client_id}
    return await make_api_request(ctx, "post", "/api/v1/query/stop", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_query_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of queries from Superset"""
    return await make_api_request(ctx, "get", "/api/v1/query/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_query_get_by_id(ctx: Context, query_id: int) -> Dict[str, Any]:
    """
    Get details for a specific query

    Args:
        query_id: ID of the query to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/query/{query_id}")


# ===== Activity and User Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_activity_get_recent(ctx: Context) -> Dict[str, Any]:
    """Get recent activity data for the current user"""
    return await make_api_request(ctx, "get", "/api/v1/log/recent_activity/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_user_get_current(ctx: Context) -> Dict[str, Any]:
    """Get information about the currently authenticated user"""
    return await make_api_request(ctx, "get", "/api/v1/me/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_user_get_roles(ctx: Context) -> Dict[str, Any]:
    """Get roles for the current user"""
    return await make_api_request(ctx, "get", "/api/v1/me/roles/")


# ===== Tag Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_list(ctx: Context) -> Dict[str, Any]:
    """Get a list of tags from Superset"""
    return await make_api_request(ctx, "get", "/api/v1/tag/")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_create(ctx: Context, name: str) -> Dict[str, Any]:
    """
    Create a new tag in Superset

    Args:
        name: Name for the tag
    """
    payload = {"name": name}
    return await make_api_request(ctx, "post", "/api/v1/tag/", data=payload)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_get_by_id(ctx: Context, tag_id: int) -> Dict[str, Any]:
    """
    Get details for a specific tag

    Args:
        tag_id: ID of the tag to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/tag/{tag_id}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_tag_objects(ctx: Context) -> Dict[str, Any]:
    """Get objects associated with tags"""
    return await make_api_request(ctx, "get", "/api/v1/tag/get_objects/")


# ===== Explore Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_form_data_create(
    ctx: Context, form_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create form data for chart exploration

    Args:
        form_data: Form data for chart configuration
    """
    return await make_api_request(
        ctx, "post", "/api/v1/explore/form_data", data=form_data
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_form_data_get(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get form data for chart exploration

    Args:
        key: Key of the form data to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/explore/form_data/{key}")


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_permalink_create(
    ctx: Context, state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a permalink for chart exploration

    Args:
        state: State data for the permalink
    """
    return await make_api_request(ctx, "post", "/api/v1/explore/permalink", data=state)


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_explore_permalink_get(ctx: Context, key: str) -> Dict[str, Any]:
    """
    Get a permalink for chart exploration

    Args:
        key: Key of the permalink to retrieve
    """
    return await make_api_request(ctx, "get", f"/api/v1/explore/permalink/{key}")


# ===== Menu Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_menu_get(ctx: Context) -> Dict[str, Any]:
    """Get the Superset menu data"""
    return await make_api_request(ctx, "get", "/api/v1/menu/")


# ===== Advanced Data Type Tools =====


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_advanced_data_type_convert(
    ctx: Context, type_name: str, value: Any
) -> Dict[str, Any]:
    """
    Convert a value to an advanced data type

    Args:
        type_name: Name of the advanced data type
        value: Value to convert
    """
    params = {
        "type_name": type_name,
        "value": value,
    }

    return await make_api_request(
        ctx, "get", "/api/v1/advanced_data_type/convert", params=params
    )


@mcp.tool()
@requires_auth
@handle_api_errors
async def superset_advanced_data_type_list(ctx: Context) -> Dict[str, Any]:
    """Get list of available advanced data types"""
    return await make_api_request(ctx, "get", "/api/v1/advanced_data_type/types")


if __name__ == "__main__":
    print("Starting Superset MCP server...")
    mcp.run()
