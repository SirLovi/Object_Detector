import React, { useState } from 'react';
import axios from 'axios';
import ReactPlayer from 'react-player';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [movementEvents, setMovementEvents] = useState([]);

  const handleFileChange = (e) => {
    setVideoFile(e.target.files[0]);
  };

  const handleAnalysis = async () => {
    if (!videoFile) {
      alert('Please select a video file.');
      return;
    }

    const formData = new FormData();
    formData.append('video', videoFile);

    try {
      const response = await axios.post('/api/analyze-video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMovementEvents(response.data.movementEvents);
    } catch (error) {
      console.error(error);
      alert('An error occurred during analysis.');
    }
  };

  return (
    <div>
      <h1>Video Analysis</h1>
      <div>
        <input type="file" accept="video/*" onChange={handleFileChange} />
        <button onClick={handleAnalysis}>Analyze</button>
      </div>
      {movementEvents.length > 0 && (
        <div>
          <h2>Movement Events</h2>
          <ul>
            {movementEvents.map((eventTime) => (
              <li key={eventTime}>{eventTime}</li>
            ))}
          </ul>
        </div>
      )}
      {videoFile && (
        <div>
          <h2>Video Player</h2>
          <ReactPlayer url={URL.createObjectURL(videoFile)} controls />
        </div>
      )}
    </div>
  );
}

export default App;
