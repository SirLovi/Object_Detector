import React, { Component } from "react";
import ReactPlayer from "react-player";
import "./App.css";
import axios from "axios";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      videoFiles: [],
      loading: false,
      videoDurations: {},
      labelColorMaps: {},
      currentTime: 0,
      activeTab: "movement",
    };
    this.videoRefs = new Map();
  }

  handleFileChange = (e) => {
    this.setState({
      videoFiles: Array.from(e.target.files).map((file) => ({
        file,
        url: URL.createObjectURL(file),
      })),
    });
  };

  setActiveTab = (tabName) => {
    this.setState({ activeTab: tabName });
  };

  handleProgress = (playedSeconds) => {
    this.setState({ currentTime: playedSeconds }, () => {
      this.forceUpdate();
    });
  };

  analyzeVideo = async (videoFile) => {
    const formData = new FormData();
    formData.append("video[]", videoFile.file);

    this.setState({ loading: true });

    try {
      const response = await axios.post(
        "http://localhost:5000/api/detect-events",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      this.setState({ loading: false });

      if (response.data.error) {
        alert(`Error in video analysis: ${response.data.error}`);
        return;
      }

      const resultFiles = response.data;
      console.log("Analysis results:", resultFiles);

      let tempVideoDurations = {};
      let tempLabelColorMaps = { ...this.state.labelColorMaps };
      const updatedVideoFiles = this.state.videoFiles.map((video) => {
        const videoData = resultFiles[video.file.name];
        if (videoData) {
          video.originalVideoWidth = videoData.original_video_width;
          video.originalVideoHeight = videoData.original_video_height;
          tempVideoDurations[video.file.name] = videoData.video_duration;
          tempLabelColorMaps[video.file.name] = videoData.label_color_map;
          return {
            ...video,
            timeline: `data:image/png;base64,${videoData.timeline}`,
            objectDetectionImagePath: `data:image/png;base64,${videoData.object_detections}`,
            objectDetections: videoData.object_detections_data,
          };
        }
        return video;
      });

      this.setState((prevState) => ({
        videoFiles: updatedVideoFiles,
        videoDurations: { ...prevState.videoDurations, ...tempVideoDurations },
        labelColorMaps: tempLabelColorMaps,
      }));
    } catch (error) {
      this.setState({ loading: false });
      alert(`An error occurred during analysis: ${error.message}`);
    }
  };

  handleTimelineClick = (videoFile, e) => {
    const videoDuration = this.state.videoDurations[videoFile.file.name];
    if (!videoDuration) return;

    const timelineElement = e.target;
    const clickX = e.nativeEvent.offsetX;
    const timelineWidth = timelineElement.clientWidth;
    const clickPositionRatio = clickX / timelineWidth;
    const clickedTime = (clickPositionRatio * videoDuration) / 1000;

    const videoRef = this.videoRefs.get(videoFile.file.name);
    if (videoRef && videoRef.current) {
      videoRef.current.seekTo(clickedTime, "seconds");
    }
  };

  handleAnalysis = async () => {
    const { videoFiles } = this.state;
    if (!videoFiles.length) {
      alert("Please select at least one video file.");
      return;
    }

    for (const videoFile of videoFiles) {
      await this.analyzeVideo(videoFile);
    }
  };

  renderVideoFile = (videoFile) => {
    const { currentTime, activeTab, videoDurations } = this.state;

    if (!this.videoRefs.has(videoFile.file.name)) {
      this.videoRefs.set(videoFile.file.name, React.createRef());
    }

    const videoDuration = videoDurations[videoFile.file.name] || 0;
    const currentPositionPercentage = (currentTime / videoDuration) * 100000;
    const isAnalyzed = videoFile.timeline || videoFile.objectDetectionImagePath;

    return (
      <div key={videoFile.file.name} className="video-container">
        <h2>{videoFile.file.name}</h2>
        <div className="video-overlay">
          <ReactPlayer
            ref={this.videoRefs.get(videoFile.file.name)}
            url={videoFile.url}
            controls
            width="100%"
            height="360px"
            onProgress={({ playedSeconds }) =>
              this.handleProgress(playedSeconds)
            }
          />
          {this.renderObjectBoxes(
            videoFile,
            this.videoRefs.get(videoFile.file.name)
          )}
        </div>

        {isAnalyzed && (
          <>
            <div className="tabs">
              <button
                onClick={() => this.setActiveTab("movement")}
                className={`tab ${activeTab === "movement" ? "active" : ""}`}
              >
                Movement Intensity
              </button>
              <button
                onClick={() => this.setActiveTab("objects")}
                className={`tab ${activeTab === "objects" ? "active" : ""}`}
              >
                Object Detection
              </button>
            </div>

            {activeTab === "movement" && videoFile.timeline && (
              <div className="timeline-container">
                <img
                  src={videoFile.timeline}
                  alt="Movement intensity timeline"
                  style={{ width: "100%" }}
                  onClick={(e) => this.handleTimelineClick(videoFile, e)}
                />
                <div
                  className="current-time-marker"
                  style={{ left: `${currentPositionPercentage}%` }}
                ></div>
              </div>
            )}

            {activeTab === "objects" && videoFile.objectDetectionImagePath && (
              <div className="timeline-container">
                <img
                  src={videoFile.objectDetectionImagePath}
                  alt="Object detection timeline"
                  style={{ width: "100%" }}
                  onClick={(e) => this.handleTimelineClick(videoFile, e)}
                />
                <div
                  className="current-time-marker"
                  style={{ left: `${currentPositionPercentage}%` }}
                ></div>
              </div>
            )}
            {this.renderColorLegend(videoFile.file.name)}
          </>
        )}
      </div>
    );
  };

  renderObjectBoxes = (videoFile, videoRef) => {
    const currentTime = this.state.currentTime;
    const objectDetections = videoFile.objectDetections || [];
    const currentDetections = objectDetections.filter(
      (detection) =>
        currentTime >= detection.time && currentTime < detection.time + 1
    );

    return currentDetections.map((detection, index) => {
      if (!videoRef.current) return null;

      const videoWidth = videoRef.current.wrapper.clientWidth;
      const videoHeight = videoRef.current.wrapper.clientHeight;
      const [x1, y1, x2, y2] = detection.box;

      const scaleX = videoWidth / videoFile.originalVideoWidth;
      const scaleY = videoHeight / videoFile.originalVideoHeight;

      const style = {
        left: `${x1 * scaleX}px`,
        top: `${y1 * scaleY}px`,
        width: `${(x2 - x1) * scaleX}px`,
        height: `${(y2 - y1) * scaleY}px`,
        borderColor:
          this.state.labelColorMaps[videoFile.file.name][detection.label],
        position: "absolute",
        zIndex: 2,
      };

      return <div key={index} className="object-box" style={style}></div>;
    });
  };

  renderColorLegend = (videoFileName) => {
    const { labelColorMaps, activeTab } = this.state;
    const labelColorMap = labelColorMaps[videoFileName];

    if (activeTab !== "objects" || !labelColorMap) {
      return null;
    }

    return (
      <div className="color-legend">
        {Object.entries(labelColorMap).map(([label, color]) => (
          <div key={label} className="legend-item">
            <span
              className="legend-color"
              style={{ backgroundColor: color }}
            ></span>
            <span className="legend-label">{label}</span>
          </div>
        ))}
      </div>
    );
  };

  render() {
    const { videoFiles, loading } = this.state;

    return (
      <div className="app">
        <h1>Video Analysis</h1>
        <div className="input-container">
          <input
            type="file"
            accept="video/*"
            onChange={this.handleFileChange}
            multiple
          />
          <button onClick={this.handleAnalysis} disabled={loading}>
            Analyze Videos
          </button>
        </div>
        {loading && <p className="loading">Analyzing videos...</p>}
        <div className="video-display">
          {videoFiles.map(this.renderVideoFile)}
        </div>
      </div>
    );
  }
}

export default App;
