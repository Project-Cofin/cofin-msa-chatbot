from rest_framework import serializers
from .models import HealthStatus as healthStatus
from .models import Chatbot as chatbot


class HealthStatusSerializer(serializers.Serializer):
    symptom = serializers.CharField()
    details = serializers.CharField()
    level = serializers.CharField()
    answer = serializers.CharField()

    class Meta:
        model = healthStatus
        fields = '__all__'


class ChatbotSerializer(serializers.Serializer):
    question = serializers.CharField()
    answer = serializers.CharField()
    label = serializers.CharField()

    class Meta:
        model = chatbot
        fields = '__all__'
