from django.db import models


class HealthStatus(models.Model):
    use_in_migrations = True
    symptom = models.TextField()
    details = models.TextField()
    level = models.TextField()
    answer = models.TextField()

    class Meta:
        db_table = "health_status"

    def __str__(self):
        return f'[{self.pk}] {self.id}'


class Chatbot(models.Model):
    use_in_migrations = True
    question = models.TextField()
    answer = models.TextField()
    label = models.IntegerField()

    class Meta:
        db_table = "chatbot"

    def __str__(self):
        return f'[{self.pk}] {self.id}'