"""Optional YouTube audio download. Requires `yt-dlp` (and ffmpeg)."""
import yt_dlp


def get_audio_from_youtube(url, output_path="data/temp_audio"):
	"""Download the audio of a YouTube video as a wav file and return its path."""
	ydl_opts = {
		'format': 'bestaudio/best',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'wav',
			'preferredquality': '192',
		}],
		'noplaylist': True,
		'outtmpl': output_path,
	}
	with yt_dlp.YoutubeDL(ydl_opts) as ydl:
		ydl.download([url])

	return output_path + ".wav"
