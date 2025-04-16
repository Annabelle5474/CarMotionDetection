from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import detect_road_yolo, frames_to_video, detect_collision_from_video
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

        # this saves annotated frames
        frames = detect_road_yolo(video_path) # no need to pass output_dir unless you want to customize it

        frames_dir = os.path.join(settings.MEDIA_ROOT, 'yolo_frames')
        video_output = os.path.join(settings.MEDIA_ROOT, 'output.mp4')

        # this func from the util
        frames_to_video(frames_dir, video_output)


        # Optionally save or display processed results
        # return render(request, 'detection/result.html', {'frames_count': len(frames)})
        # return render(request, 'detection/result.html', {
        #     'frames': frames,
        #     'frames_count': len(frames)
        # })

        # Video URL for template
        video_url = settings.MEDIA_URL + 'output.mp4'

        # this is in the html data
        return render(request, 'detection/result.html', {
            'frames_count': len(frames),
            'video_url': video_url
        })
    
    return render(request, 'detection/upload.html')

@csrf_exempt
def upload_video2(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video.name, video)
        video_path = fs.path(filename)

        output_dir = os.path.join(settings.MEDIA_ROOT, 'collision_frames')
        detect_collision_from_video(video_path, output_dir)

        frame_list = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
        frame_urls = [os.path.join(settings.MEDIA_URL, 'collision_frames', f) for f in frame_list]

        return render(request, 'detection/result.html', {
            'frames': frame_urls,
            'frames_count': len(frame_urls),
        })
    return render(request, 'detection/upload.html')