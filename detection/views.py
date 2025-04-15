from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import detect_road
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os

# In Django (and other Python projects), a utils.py file is used to store utility functions — 
# reusable helper code that doesn't belong to models, views, or templates.

@csrf_exempt
def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        video_path = fs.path(filename)

        frames = detect_road(video_path) # no need to pass output_dir unless you want to customize it

        # Optionally save or display processed results
        # return render(request, 'detection/result.html', {'frames_count': len(frames)})
        return render(request, 'detection/result.html', {
            'frames': frames,
            'frames_count': len(frames)
        })
    
    return render(request, 'detection/upload.html')

