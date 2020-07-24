# Deploy deepgalaxy project with Django

This project is a Django web applicationn for DeepGalaxy project, consists of several main components:

- Image uplolad and display [djangoimageupload](https://learndjango.com/tutorials/django-file-and-image-uploads-tutorial)

- Machine learning model deployment [deploymachinelearning.com](https://deploymachinelearning.com)

- Polished frontend user interaction 

- It contains tests for both ML code and server code


This application is (going to be) hosted in one of the main cloud provider, in order to provide public access.

## The code structure

In the `research` directory there are:

- code for training machine learning models (TBD).

In the `backend` directory there is Django application. Under /apps there are:

- endpoints: Django Rest application

- ml: deepgalaxy model code and training dataset


In the `docker` directory there are dockerfiles for running the service in the container.

## How to use

Install dependencies from `requirements.txt`

To clean historic data, you can

```
>>> rm db.sqlite3
>>> python manage.py makemigrations
>>> python manage.py migrate
```

To run the server, you can

```python manage.py runserver```

To test the code and endpoint (TBD)

```
>>> python manage.py test apps.ml.tests
>>> python manage.py test apps.endpoints.tests
>>> python manage.py test apps
```

Open your browser and go to `0.0.0.0:8000/post/` to start, have fun! ;) 
