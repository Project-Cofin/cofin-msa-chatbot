from rest_framework import serializers
# pip install Django django-rest-framework
from .models import HealthStatus as healthStatus


class HealthStatusSerializer(serializers.Serializer):
    symptom = serializers.CharField()
    details = serializers.CharField()
    level = serializers.CharField()
    answer = serializers.CharField()

    class Meta:
        model = healthStatus
        fields = '__all__'
