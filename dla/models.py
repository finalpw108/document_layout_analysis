from django.db import models

class DataUpload(models.Model):
    email=models.CharField(max_length=40)
    file=models.FileField(upload_to="../../media/%y")

