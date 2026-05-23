import streamlit as st
import tempfile
import os
import subprocess
import sys
import zipfile
import io
import math
import time
from datetime import datetime
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Video Transcription App",
    page_icon="🎥", 
    layout="wide"
)

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def extract_audio_ffmpeg(video_path):
    """Extract audio using FFmpeg"""
    try:
        # Create temp audio file
        audio_path = tempfile.mktemp(suffix='.wav')
        
        # Get duration
        duration_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', video_path]
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        # Extract audio
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', audio_path]
        subprocess.run(cmd, capture_output=True, check=True)
        
        return audio_path, duration
    except Exception as e:
        return None, None

def extract_audio_moviepy(video_path):
    """Extract audio using MoviePy as fallback"""
    try:
        # Try to install moviepy if needed
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            st.info("📦 Installing MoviePy...")
            if install_package("moviepy"):
                from moviepy.editor import VideoFileClip
            else:
                return None, None
        
        # Extract audio
        audio_path = tempfile.mktemp(suffix='.wav')
        video = VideoFileClip(video_path)
        duration = video.duration
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        return audio_path, duration
    except Exception as e:
        return None, None

def split_audio_chunks(audio_path, duration, chunk_duration=30):
    """Split audio into chunks"""
    chunks = []
    num_chunks = math.ceil(duration / chunk_duration)
    
    os.makedirs("audio_chunks", exist_ok=True)
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_file = f"audio_chunks/chunk_{i+1:03d}.wav"
        
        if check_ffmpeg():
            # Use FFmpeg to split
            cmd = ['ffmpeg', '-i', audio_path, '-ss', str(start_time), '-t', str(chunk_duration), '-y', chunk_file]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                end_time = min(start_time + chunk_duration, duration)
                
                chunks.append({
                    'chunk_number': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'file_path': chunk_file,
                    'file_size': os.path.getsize(chunk_file)
                })
            except:
                continue
        else:
            # Try with pydub as fallback
            try:
                if not 'AudioSegment' in globals():
                    try:
                        from pydub import AudioSegment
                    except ImportError:
                        install_package("pydub")
                        from pydub import AudioSegment
                
                audio = AudioSegment.from_wav(audio_path)
                start_ms = int(start_time * 1000)
                end_ms = int(min((start_time + chunk_duration) * 1000, len(audio)))
                
                chunk = audio[start_ms:end_ms]
                chunk.export(chunk_file, format="wav")
                
                chunks.append({
                    'chunk_number': i + 1,
                    'start_time': start_time,
                    'end_time': end_ms / 1000,
                    'duration': (end_ms - start_ms) / 1000,
                    'file_path': chunk_file,
                    'file_size': os.path.getsize(chunk_file)
                })
            except:
                continue
    
    return chunks

def format_time(seconds):
    """Format seconds as MM:SS"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def create_templates(chunks):
    """Create transcription templates"""
    os.makedirs("templates", exist_ok=True)
    templates = []
    
    for chunk in chunks:
        content = f"""TRANSCRIPTION TEMPLATE - CHUNK {chunk['chunk_number']:03d}
{'='*60}
Time Range: {format_time(chunk['start_time'])} - {format_time(chunk['end_time'])}
Duration: {chunk['duration']:.1f} seconds
Audio File: {chunk['file_path']}
{'='*60}

INSTRUCTIONS:
1. Play the audio file above
2. Type what you hear below
3. Replace [TRANSCRIPTION NEEDED] with actual text

TRANSCRIPTION:
[TRANSCRIPTION NEEDED - Replace this with the spoken content]

NOTES:
- Include all speech, even "um", "uh"
- Use [unclear] for inaudible parts
- Use [music] for background music
- Indicate speakers: "Speaker 1:", "Speaker 2:"
"""
        
        template_file = f"templates/chunk_{chunk['chunk_number']:03d}_template.txt"
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(content)
        templates.append(template_file)
    
    return templates

def transcribe_audio_chunks(chunks):
    """Transcribe audio chunks using OpenAI Whisper"""
    try:
        import whisper
    except ImportError:
        st.info("📦 Installing OpenAI Whisper...")
        if not install_package("openai-whisper"):
            st.error("❌ Failed to install Whisper. Please install manually: pip install openai-whisper")
            return None
        
        try:
            import whisper
        except ImportError:
            st.error("❌ Whisper import failed. Please restart the app after installation.")
            return None
    
    # Load model (using base model for speed/accuracy balance)
    try:
        model = whisper.load_model("base")
    except Exception as e:
        st.error(f"❌ Failed to load Whisper model: {e}")
        return None
    
    transcriptions = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"🎙️ Transcribing chunk {i+1}/{len(chunks)}...")
        
        try:
            result = model.transcribe(chunk['file_path'])
            transcription = result['text'].strip()
            
            transcriptions.append({
                'chunk_number': chunk['chunk_number'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'transcription': transcription,
                'confidence': result.get('confidence', 0)
            })
            
        except Exception as e:
            transcriptions.append({
                'chunk_number': chunk['chunk_number'],
                'start_time': chunk['start_time'],
                'end_time': chunk['end_time'],
                'transcription': f"[TRANSCRIPTION FAILED: {str(e)}]",
                'confidence': 0
            })
        
        progress_bar.progress((i + 1) / len(chunks))
    
    progress_bar.empty()
    status_text.empty()
    
    return transcriptions

def create_download_kit(chunks, templates, metadata, transcriptions=None):
    """Create downloadable ZIP kit"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add instructions
        instructions = f"""VIDEO TRANSCRIPTION PROJECT
{'='*50}
Video: {metadata['filename']}
Duration: {format_time(metadata['duration'])}
Chunks: {len(chunks)} x {metadata['chunk_duration']}s each
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AUTOMATIC TRANSCRIPTION:
1. Install: pip install openai-whisper
2. Run: whisper audio_chunks/*.wav --output_format txt

MANUAL TRANSCRIPTION:
1. Play each audio file in audio_chunks/
2. Fill corresponding template in templates/
3. Replace [TRANSCRIPTION NEEDED] with actual text

CHUNK LIST:
"""
        for chunk in chunks:
            instructions += f"\n{chunk['chunk_number']:03d}: {format_time(chunk['start_time'])}-{format_time(chunk['end_time'])} | {chunk['file_path']}"
        
        zf.writestr("README.txt", instructions)
        
        # Add audio chunks
        for chunk in chunks:
            if os.path.exists(chunk['file_path']):
                zf.write(chunk['file_path'], chunk['file_path'])
        
        # Add templates
        for template in templates:
            if os.path.exists(template):
                zf.write(template, template)
        
        # Add transcriptions if available
        if transcriptions:
            os.makedirs("transcriptions", exist_ok=True)
            
            # Create full transcription file
            full_transcript = f"COMPLETE TRANSCRIPTION - {metadata['filename']}\n{'='*60}\n\n"
            
            for trans in transcriptions:
                full_transcript += f"[{format_time(trans['start_time'])} - {format_time(trans['end_time'])}]\n"
                full_transcript += f"{trans['transcription']}\n\n"
            
            zf.writestr("transcriptions/full_transcription.txt", full_transcript)
            
            # Create individual transcription files as 0.txt, 30.txt, 60.txt, etc. (time-based naming)
            for trans in transcriptions:
                # Calculate time-based filename (start time of chunk)
                start_time_seconds = (trans['chunk_number'] - 1) * metadata['chunk_duration']
                transcription_file = f"transcriptions/{start_time_seconds}.txt"
                
                # Save transcription text only to individual files
                with open(transcription_file, 'w', encoding='utf-8') as f:
                    f.write(trans['transcription'])
                
                # Add to ZIP file
                zf.write(transcription_file, f"transcriptions/{start_time_seconds}.txt")
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.title("🎥 Video Transcription App")
    st.markdown("**Convert videos to text in manageable 30-second chunks!**")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    chunk_duration = st.sidebar.slider("Chunk Duration (seconds)", 15, 60, 30, 5)
    
    # System status
    st.sidebar.header("🔧 System Status")
    ffmpeg_available = check_ffmpeg()
    
    if ffmpeg_available:
        st.sidebar.success("✅ FFmpeg available")
    else:
        st.sidebar.error("❌ FFmpeg not found")
        st.sidebar.markdown("""
        **Install FFmpeg:**
        1. Download: https://ffmpeg.org/
        2. Add to PATH
        3. Restart app
        """)
    
    # Main content
    st.header("📁 Upload Video File")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'wmv'],
        help="Supported: MP4, MOV, AVI, MKV, WMV"
    )
    
    if uploaded_file:
        file_size = uploaded_file.size / 1024 / 1024
        st.success(f"✅ File ready: **{uploaded_file.name}** ({file_size:.1f} MB)")
        
        if not ffmpeg_available:
            st.error("❌ Cannot process without FFmpeg. Please install FFmpeg first.")
            return
        
        if st.button("🚀 Process Video", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            start_time = time.time()
            
            try:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_video:
                    temp_video.write(uploaded_file.read())
                    video_path = temp_video.name
                
                progress.progress(0.1)
                status.info("📁 Video file prepared")
                
                # Extract audio
                status.info("🎵 Extracting audio...")
                audio_path, duration = extract_audio_ffmpeg(video_path)
                
                if not audio_path:
                    st.error("❌ Failed to extract audio")
                    return
                
                progress.progress(0.4)
                status.info(f"✅ Audio extracted - Duration: {format_time(duration)}")
                
                # Split into chunks
                status.info("✂️ Creating audio chunks...")
                chunks = split_audio_chunks(audio_path, duration, chunk_duration)
                
                if not chunks:
                    st.error("❌ Failed to create chunks")
                    return
                
                progress.progress(0.7)
                status.info(f"✅ Created {len(chunks)} audio chunks")
                
                # Create templates
                status.info("📝 Creating transcription templates...")
                templates = create_templates(chunks)
                
                progress.progress(0.9)
                
                # Transcribe audio chunks
                status.info("🎙️ Transcribing audio chunks...")
                transcriptions = transcribe_audio_chunks(chunks)
                
                if transcriptions:
                    status.info("✅ Transcription completed!")
                else:
                    status.info("⚠️ Transcription skipped (Whisper not available)")
                
                # Create download kit
                status.info("📦 Preparing download kit...")
                metadata = {
                    'filename': uploaded_file.name,
                    'duration': duration,
                    'chunk_duration': chunk_duration
                }
                
                kit_data = create_download_kit(chunks, templates, metadata, transcriptions)
                
                # Store results
                st.session_state.results = {
                    'chunks': chunks,
                    'templates': templates,
                    'transcriptions': transcriptions,
                    'metadata': metadata,
                    'kit_data': kit_data
                }
                
                progress.progress(1.0)
                elapsed = time.time() - start_time
                status.success(f"🎉 Completed in {elapsed:.1f} seconds!")
                
                # Cleanup
                os.unlink(video_path)
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

# Show results
if 'results' in st.session_state:
    results = st.session_state.results
    chunks = results['chunks']
    metadata = results['metadata']
    
    st.header("📊 Processing Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Video Duration", format_time(metadata['duration']))
    with col2:
        st.metric("Audio Chunks", len(chunks))
    with col3:
        st.metric("Chunk Size", f"{metadata['chunk_duration']}s")
    with col4:
        total_size = sum(chunk['file_size'] for chunk in chunks) / 1024 / 1024
        st.metric("Total Audio", f"{total_size:.1f} MB")
    
    # Download section
    st.header("📥 Download Transcription Kit")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.download_button(
            "📦 Download Complete Kit (Audio + Templates)",
            results['kit_data'],
            f"transcription_kit_{Path(metadata['filename']).stem}.zip",
            "application/zip",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        kit_contents = f"""
        **Kit Contents:**
        • {len(chunks)} audio chunks
        • {len(chunks)} text templates"""
        
        if results.get('transcriptions'):
            kit_contents += f"\n• {len(results['transcriptions'])} transcriptions"
        
        kit_contents += "\n• Instructions & setup guide"
        
        st.info(kit_contents)
    
    # Transcription options
    st.header("🎙️ Transcription Options")
    
    tab1, tab2 = st.tabs(["🤖 Automatic (Whisper)", "✍️ Manual"])
    
    with tab1:
        if results.get('transcriptions'):
            st.success("✅ **Automatic transcription completed!**")
            st.markdown("**Transcriptions are included in your download kit.**")
        else:
            st.markdown("""
            **Using OpenAI Whisper (Recommended):**
            
            1. **Install Whisper:**
            ```bash
            pip install openai-whisper
            ```
            
            2. **Transcribe all chunks:**
            ```bash
            whisper audio_chunks/*.wav --output_format txt --model base
            ```
            
            3. **Or transcribe individually:**
            ```bash
            whisper audio_chunks/chunk_001.wav --output_format txt
            ```
            
            **Model Options:**
            - `tiny` - Fastest, least accurate
            - `base` - Good balance (recommended)
            - `small` - Better accuracy
            - `medium` - High accuracy
            - `large` - Best accuracy, slowest
            """)
    
    with tab2:
        st.markdown("""
        **Manual Transcription Steps:**
        
        1. **Download the kit** above
        2. **Extract the ZIP file**
        3. **For each audio chunk:**
           - Play `audio_chunks/chunk_XXX.wav`
           - Open `templates/chunk_XXX_template.txt`
           - Replace `[TRANSCRIPTION NEEDED]` with what you hear
           - Save the file
        
        **Tips:**
        - Use good headphones
        - Play chunks multiple times if needed
        - Include all speech (even "um", "uh")
        - Mark unclear parts as `[unclear]`
        - Note speaker changes: `Speaker 1:`, `Speaker 2:`
        """)
    
    # Display transcriptions if available
    if results.get('transcriptions'):
        st.header("📝 Transcriptions")
        
        # Full transcription view
        with st.expander("📄 Full Transcription", expanded=True):
            full_text = ""
            for trans in results['transcriptions']:
                full_text += f"**[{format_time(trans['start_time'])} - {format_time(trans['end_time'])}]**\n"
                full_text += f"{trans['transcription']}\n\n"
            
            st.markdown(full_text)
        
        # Individual chunks
        st.subheader("🎬 Individual Chunks")
        
        for trans in results['transcriptions'][:5]:  # Show first 5
            with st.expander(f"Chunk {trans['chunk_number']} ({format_time(trans['start_time'])} - {format_time(trans['end_time'])})"):
                st.write(trans['transcription'])
        
        if len(results['transcriptions']) > 5:
            st.info(f"📄 Showing 5 of {len(results['transcriptions'])} chunks. Download the complete kit for all transcriptions.")
    
    # Preview chunks
    st.header("🎵 Audio Chunks Preview")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        with st.expander(f"🎬 Chunk {chunk['chunk_number']} ({format_time(chunk['start_time'])} - {format_time(chunk['end_time'])})"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Duration:** {chunk['duration']:.1f} seconds")
                st.write(f"**File Size:** {chunk['file_size']/1024:.1f} KB")
            
            with col2:
                if os.path.exists(chunk['file_path']):
                    with open(chunk['file_path'], 'rb') as f:
                        st.download_button(
                            f"💾 Download Audio Chunk",
                            f.read(),
                            f"chunk_{chunk['chunk_number']:03d}.wav",
                            "audio/wav",
                            key=f"audio_download_{i}"
                        )
    
    if len(chunks) > 3:
        st.info(f"📄 Showing 3 of {len(chunks)} chunks. Download the complete kit to get all chunks.")

def format_time(seconds):
    """Format seconds as MM:SS"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def main():
    st.title("🎥 Video Transcription App")
    st.markdown("**Convert your videos to text in manageable 30-second chunks!**")
    
    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    chunk_duration = st.sidebar.slider("Chunk Duration (seconds)", 15, 60, 30, 5)
    
    # System check
    st.sidebar.header("🔧 System Status")
    ffmpeg_available = check_ffmpeg()
    
    if ffmpeg_available:
        st.sidebar.success("✅ FFmpeg available")
    else:
        st.sidebar.error("❌ FFmpeg required")
        st.sidebar.markdown("""
        **Install FFmpeg:**
        1. Visit: https://ffmpeg.org/download.html
        2. Download for Windows
        3. Add to system PATH
        4. Restart this app
        """)
    
    # Main content
    st.header("📁 Upload Your Video")
    
    if not ffmpeg_available:
        st.error("⚠️ FFmpeg is required for video processing. Please install FFmpeg first.")
        st.stop()
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv', 'wmv'],
        help="Supported formats: MP4, MOV, AVI, MKV, WMV"
    )
    
    if uploaded_file:
        file_size = uploaded_file.size / 1024 / 1024
        st.success(f"✅ **{uploaded_file.name}** uploaded ({file_size:.1f} MB)")
        
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            start_time = time.time()
            
            try:
                # Save video temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_video:
                    temp_video.write(uploaded_file.read())
                    video_path = temp_video.name
                
                progress.progress(0.1)
                status.info("📁 Video file saved")
                
                # Extract audio
                status.info("🎵 Extracting audio with FFmpeg...")
                audio_path, duration = extract_audio_ffmpeg(video_path)
                
                if not audio_path:
                    # Try MoviePy as fallback
                    status.info("🔄 Trying alternative method...")
                    audio_path, duration = extract_audio_moviepy(video_path)
                
                if not audio_path:
                    st.error("❌ Failed to extract audio with all methods")
                    return
                
                progress.progress(0.4)
                status.info(f"✅ Audio extracted - Duration: {format_time(duration)}")
                
                # Split into chunks
                status.info("✂️ Creating audio chunks...")
                chunks = split_audio_chunks(audio_path, duration, chunk_duration)
                
                if not chunks:
                    st.error("❌ Failed to create audio chunks")
                    return
                
                progress.progress(0.7)
                status.info(f"✅ Created {len(chunks)} audio chunks")
                
                # Create templates
                status.info("📝 Creating transcription templates...")
                templates = create_templates(chunks)
                
                progress.progress(0.9)
                
                # Transcribe audio chunks
                status.info("🎙️ Transcribing audio chunks...")
                transcriptions = transcribe_audio_chunks(chunks)
                
                if transcriptions:
                    status.info("✅ Transcription completed!")
                else:
                    status.info("⚠️ Transcription skipped (Whisper not available)")
                
                # Create download kit
                status.info("📦 Preparing download kit...")
                metadata = {
                    'filename': uploaded_file.name,
                    'duration': duration,
                    'chunk_duration': chunk_duration
                }
                
                kit_data = create_download_kit(chunks, templates, metadata, transcriptions)
                
                # Store results
                st.session_state.processing_results = {
                    'chunks': chunks,
                    'templates': templates,
                    'transcriptions': transcriptions,
                    'metadata': metadata,
                    'kit_data': kit_data
                }
                
                progress.progress(1.0)
                elapsed = time.time() - start_time
                status.success(f"🎉 Processing completed in {elapsed:.1f} seconds!")
                
                # Cleanup temp files
                os.unlink(video_path)
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                st.exception(e)

# Display results if available
if 'processing_results' in st.session_state:
    results = st.session_state.processing_results
    chunks = results['chunks']
    metadata = results['metadata']
    
    st.header("📊 Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Video Duration", format_time(metadata['duration']))
    with col2:
        st.metric("Audio Chunks", len(chunks))
    with col3:
        st.metric("Chunk Duration", f"{metadata['chunk_duration']}s")
    with col4:
        total_size = sum(chunk['file_size'] for chunk in chunks) / 1024 / 1024
        st.metric("Total Audio Size", f"{total_size:.1f} MB")
    
    # Download section
    st.header("📥 Download Complete Kit")
    
    st.download_button(
        "📦 Download Transcription Kit",
        results['kit_data'],
        f"transcription_kit_{Path(metadata['filename']).stem}.zip",
        "application/zip",
        type="primary",
        use_container_width=True,
        help="Contains audio chunks, templates, and transcriptions"
    )
    
    # Display transcriptions if available
    if results.get('transcriptions'):
        st.header("📝 Transcriptions")
        
        # Full transcription view
        with st.expander("📄 Full Transcription", expanded=True):
            full_text = ""
            for trans in results['transcriptions']:
                full_text += f"**[{format_time(trans['start_time'])} - {format_time(trans['end_time'])}]**\n"
                full_text += f"{trans['transcription']}\n\n"
            
            st.markdown(full_text)
        
        # Individual chunks
        st.subheader("🎬 Individual Chunks")
        
        for trans in results['transcriptions'][:5]:  # Show first 5
            with st.expander(f"Chunk {trans['chunk_number']} ({format_time(trans['start_time'])} - {format_time(trans['end_time'])})"):
                st.write(trans['transcription'])
        
        if len(results['transcriptions']) > 5:
            st.info(f"📄 Showing 5 of {len(results['transcriptions'])} chunks. Download the complete kit for all transcriptions.")
    
    # Quick preview
    st.header("🎵 Audio Chunks Preview")
    
    for chunk in chunks[:3]:  # Show first 3
        with st.expander(f"Chunk {chunk['chunk_number']} ({format_time(chunk['start_time'])} - {format_time(chunk['end_time'])})"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Duration:** {chunk['duration']:.1f}s")
                st.write(f"**Size:** {chunk['file_size']/1024:.1f} KB")
            
            with col2:
                if os.path.exists(chunk['file_path']):
                    with open(chunk['file_path'], 'rb') as f:
                        st.download_button(
                            "💾 Download",
                            f.read(),
                            f"chunk_{chunk['chunk_number']:03d}.wav",
                            "audio/wav",
                            key=f"chunk_download_{chunk['chunk_number']}"
                        )
    
    if len(chunks) > 3:
        st.info(f"+ {len(chunks) - 3} more chunks in the complete download kit")
    
    # Instructions
    st.header("📋 Next Steps")
    
    st.success("""
    **Your transcription kit is ready!**
    
    1. **Download the kit** above
    2. **Choose your transcription method:**
       - **Automatic:** Install Whisper and run the provided commands
       - **Manual:** Use the template files and audio chunks
    3. **Follow the detailed instructions** in the downloaded README.txt
    """)

if __name__ == "__main__":
    main()
