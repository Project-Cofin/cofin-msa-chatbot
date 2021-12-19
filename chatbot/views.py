from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from chatbot.models_data import DbUploader
from chatbot.models import HealthStatus
from chatbot.serializer import HealthStatusSerializer


@api_view(['GET'])
@parser_classes([JSONParser])
def upload(request):
    print('############ 1 ##########')
    DbUploader().insert_data()
    return JsonResponse({'Chatbot Data Upload': 'SUCCESS'})


@api_view(['POST'])
@parser_classes([JSONParser])
def find_by_detail(request):
    print('############ 2 ##########')
    quest = request.data
    print(quest)
    answer = HealthStatus.objects\
        .filter(symptom=quest['symptom'], details=quest['details']).get()
        # .only('symptom', 'level', 'answer')
    serializer = HealthStatusSerializer(answer)
    return JsonResponse(data=serializer.data, safe=False)


@api_view(['GET'])
@parser_classes([JSONParser])
def find_all(request):
    print('############ 3 ##########')
    answer = HealthStatus.objects.raw("select * from health_status group by symptom")
    serializer = HealthStatusSerializer(answer, many=True)
    return JsonResponse(data=serializer.data, safe=False)