from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.db import models

class Images(models.Model):
    image = models.ImageField(upload_to="images/")

@receiver(post_delete, sender=Images)
def submission_delete(sender, instance, **kwargs):
    instance.image.delete(False) 