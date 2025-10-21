from fastapi import Header, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class SimpleTokenAuth:
    def __init__(self, token: str):
        self.token = token

    async def __call__(self, request: Request):
        auth = request.headers.get("authorization")
        if not auth:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        if auth.strip() != f"Bearer {self.token}":
            raise HTTPException(status_code=403, detail="Invalid token")
