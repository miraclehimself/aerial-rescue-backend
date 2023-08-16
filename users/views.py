from rest_framework.views import APIView
from .serializers import UserSerializer, LoginSerializer
from .models import User
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.response import Response
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken, AccessToken
from rest_framework.permissions import IsAuthenticated
import jwt, datetime

# Create your views here.
class RegisterView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        data = serializer.data
        user = authenticate(email= serializer.data['email'], password= request.data['password'])
        refresh = RefreshToken.for_user(user)
        return Response({
             "message": "Regisration Successfull",
             'data': data,
             'token': str(refresh.access_token),
             'refresh': str(refresh),
        })
    

class LoginView(APIView):
    def post(self, request):
        # try:
        #     # serializer = LoginSerializer(data=request.data)
        #     # return Response(serializer.is_valid(raise_exception=True))
            
                email = request.data['email']
                password = request.data['password']
                user= User.objects.filter(email=email).first()

                if user is None:
                     return Response({
                        'status': 400,
                        'message': 'User not Founf',
                        'data' : {}
                    }, 404)
                user = authenticate(email= email, password= password)
                if user is None:
                    return Response({
                        'status': 401,
                        'message': 'Incorrect credentials',
                        'data' : {}
                    }, 401)
                refresh = RefreshToken.for_user(user)

                return Response({
                    'status': 200,
                    'message': 'Authenicated Successfully',
                    'token': str(refresh.access_token),
                    'refresh': str(refresh),
                })

class GetUser(APIView):
    permission_classes = (IsAuthenticated,)
    serializer_class = UserSerializer

    def get(self, request, *args, **kwargs):
         serializer = self.serializer_class(request.user)
         return Response({
              'message': 'User Retrieved Successfully',
              'data': serializer.data,
              
              }, 200)
    

        



        # payload = {
        #     'id': user.id,
        #     'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7),
        #     'iat': datetime.datetime.utcnow()
        # }
        # token = jwt.encode(payload, 'secret', algorithm='HS256')
        # response = Response()
        # response.set_cookie(key='jwt', value=token, httponly=True)
        # response.data = {
        #     'message': 'Authentication Successful',
        #     'token': token
        # }
        # return response
    
# class UserView(APIView):
#     def get(self, request):
        
