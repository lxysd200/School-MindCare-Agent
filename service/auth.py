from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from Entities.entities import UserAccount
from service.db import get_db_session
from schema.dtos import RegisterRequest


bearer_scheme = HTTPBearer(auto_error=False)


class AuthService:
    def __init__(self) -> None:
        self.secret_key = (os.getenv("JWT_SECRET_KEY") or "").strip() or "change-me-in-env"
        self.algorithm = "HS256"
        self.expire_minutes = int((os.getenv("JWT_EXPIRE_MINUTES") or "120").strip())

    def create_access_token(self, user: UserAccount) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "display_name": user.display_name,
            "roles": user.roles,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=self.expire_minutes)).timestamp()),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 已过期，请重新登录",
            ) from exc
        except jwt.InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的 Token",
            ) from exc

    def authenticate_user(self, username: str, password: str) -> UserAccount:
        db = get_db_session()
        try:
            user = db.query(UserAccount).filter(UserAccount.username == username).first()
            if user is None or not verify_password(password, user.password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户名或密码错误",
                )
            return user
        finally:
            db.close()

    def register_user(self, request: RegisterRequest) -> UserAccount:
        db = get_db_session()
        try:
            existing_user = db.query(UserAccount).filter(UserAccount.username == request.username).first()
            if existing_user is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="用户名已存在",
                )

            display_name = (request.displayName or "").strip() or request.username
            user = UserAccount(
                username=request.username.strip(),
                display_name=display_name,
                password_hash=hash_password(request.password),
                roles_csv="ROLE_USER",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> UserAccount:
        db = get_db_session()
        try:
            user = db.query(UserAccount).filter(UserAccount.id == user_id).first()
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户不存在或已失效",
                )
            return user
        finally:
            db.close()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{base64.b64encode(salt).decode('utf-8')}${base64.b64encode(password_hash).decode('utf-8')}"


def verify_password(password: str, stored_password_hash: str) -> bool:
    if "$" not in stored_password_hash:
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy_hash, stored_password_hash)

    salt_b64, password_hash_b64 = stored_password_hash.split("$", 1)
    salt = base64.b64decode(salt_b64.encode("utf-8"))
    expected_hash = base64.b64decode(password_hash_b64.encode("utf-8"))
    candidate_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(candidate_hash, expected_hash)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> UserAccount:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization Bearer Token",
        )

    auth_service = AuthService()
    payload = auth_service.decode_token(credentials.credentials)
    subject = payload.get("sub")
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 缺少用户标识",
        )

    try:
        user_id = int(subject)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 用户标识无效",
        ) from exc

    return auth_service.get_user_by_id(user_id)
