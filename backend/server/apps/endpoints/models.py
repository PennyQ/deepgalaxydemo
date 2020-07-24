from django.db import models

class Post(models.Model):
    title = models.TextField()
    cover = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.title


class Home(models.Model):
    image = models.ImageField(upload_to='prediction/')

    # def get_remote_image(self):
    #     if not self.image_file:
    #         img_temp = NamedTemporaryFile(delete=True)
    #         img_temp.write(urlopen(self.image_url).read())
    #         img_temp.flush()
    #         self.image_file.save(f"image_{self.pk}", File(img_temp))
    #     self.save()
# layout = Home()
# layout.image = "prediction/*.png"
# layout.save()