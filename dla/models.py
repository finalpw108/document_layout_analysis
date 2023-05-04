from django.db import models

class UsersDataUpload(models.Model):
    username=models.CharField(max_length=40)
    # email=models.EmailField(max_length=40,default="some")
    file=models.FileField(upload_to="pdf")
    json=models.CharField(max_length=200,default="default")
    html=models.CharField(max_length=200,default="default")

    # def __str__(self):
    #     return self.username

