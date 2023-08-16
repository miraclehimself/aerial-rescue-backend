from rest_framework import serializers
from .models import ImageModel

class ImageSerializer(serializers.ModelSerializer):

    user_id = serializers.ReadOnlyField(source = 'user.id')
    inputted_image = serializers.ImageField(required=True)
    # inputted_image = serializers.ImageField(required=True)
    

    class Meta:
        model = ImageModel
        fields = ('id', 'user_id', 'identify_object', 'inputted_image', 'outputted_image', 'result')

        read_only_fields = ('id', 'user_id')
        required_fields = ('user_id', 'inputted_image', 'identify_object')
        
 