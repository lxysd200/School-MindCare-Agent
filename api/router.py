from fastapi import APIRouter, Depends
from Entities.entities import UserAccount
from schema.dtos import ChatRequest, LoginRequest, LoginResponse, RegisterRequest, UserProfileResponse
from service.auth import AuthService, get_current_user
from service.chat import ChatService
from fastapi.responses import StreamingResponse

router = APIRouter()
auth_service = AuthService()


@router.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = auth_service.authenticate_user(request.username, request.password)
    access_token = auth_service.create_access_token(user)
    return LoginResponse(
        accessToken=access_token,
        expiresIn=auth_service.expire_minutes * 60,
        user=UserProfileResponse(
            id=user.id,
            username=user.username,
            displayName=user.display_name,
            roles=user.roles,
        ),
    )


@router.post("/api/auth/register", response_model=LoginResponse)
async def register(request: RegisterRequest):
    user = auth_service.register_user(request)
    access_token = auth_service.create_access_token(user)
    return LoginResponse(
        accessToken=access_token,
        expiresIn=auth_service.expire_minutes * 60,
        user=UserProfileResponse(
            id=user.id,
            username=user.username,
            displayName=user.display_name,
            roles=user.roles,
        ),
    )


@router.get("/api/auth/me", response_model=UserProfileResponse)
async def me(current_user: UserAccount = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        displayName=current_user.display_name,
        roles=current_user.roles,
    )

@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, current_user: UserAccount = Depends(get_current_user)):
    service = ChatService()
    return StreamingResponse(service.streamChat(request, current_user), media_type="text/event-stream")

@router.get("/api/chat/history")
async def chat_history(current_user: UserAccount = Depends(get_current_user)):
    service = ChatService()
    return service.get_chat_history(current_user)

@router.get("/api/chat/conversation/{publicId}")
async def chat_conversation(publicId: str, current_user: UserAccount = Depends(get_current_user)):
    service = ChatService()
    return service.get_conversation(publicId, current_user)

    
