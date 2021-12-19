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


# class Chatbot(models.Model):
#     use_in_migrations = True
#     type = models.CharField(max_length=10)
#     name = models.TextField()
#     meta = models.TextField()
#     latitude = models.FloatField(null=True)
#     longitude = models.FloatField(null=True)
#     population = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(30)], null=True)
#     cases = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(20)], null=True)
#
#     class Meta:
#         db_table = "chatbot"
#
#     def __str__(self):
#         return f'[{self.pk}] {self.id}'