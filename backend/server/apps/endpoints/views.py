from rest_framework import viewsets
from rest_framework import mixins

import json
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response

from django.views.generic import CreateView, ListView
from django.urls import reverse_lazy
from .models import Post, Home
from .forms import PostForm

from apps.ml.classifier.deepgalaxy import DeepGalaxyClassifier
from django.http import HttpResponse, HttpResponseRedirect

class CreatePostView(CreateView): # new
    model = Post
    form_class = PostForm
    template_name = 'post.html'
    success_url = reverse_lazy('home')

    def form_valid(self, form):
        self.object = form.save(commit=False)
        self.object.user = self.request.user
        self.object.save()
        # trigger deepgalaxy code
        my_alg = DeepGalaxyClassifier()
        response = my_alg.compute_prediction('/Users/pennyqxr/Code/deepgalaxydemo/backend/server/media/images/*.jpg')
        return HttpResponseRedirect(self.get_success_url())



class HomePageView(ListView):
    context_object_name = 'context'
    template_name = 'home.html'
    queryset = Home.objects.all()

    def get_context_data(self, **kwargs):
        context = super(HomePageView, self).get_context_data(**kwargs)
        context['post'] = Post.objects.all()
        return context