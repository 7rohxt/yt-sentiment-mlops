// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  
  let API_KEY = null;
  try {
    const res = await fetch('http://127.0.0.1:5000/get_youtube_key');
    const data = await res.json();
    API_KEY = data.api_key;
  } catch (err) {
    console.error("Failed to load YouTube API key:", err);
    outputDiv.innerHTML = "<p>Error loading YouTube API key. Please check backend.</p>";
    return;
  }

  const API_URL = 'http://127.0.0.1:5000/'; // Flask backend

  // Get the current tab’s URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (!match || !match[1]) {
      outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
      return;
    }

    const videoId = match[1];
    outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p>Fetching comments...</p>`;

    const comments = await fetchComments(videoId);
    if (comments.length === 0) {
      outputDiv.innerHTML += "<p>No comments found for this video.</p>";
      return;
    }

    outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;
    const predictions = await getSentimentPredictions(comments);

    if (!predictions) {
      outputDiv.innerHTML += "<p>Error fetching predictions.</p>";
      return;
    }

    // --- Sentiment metrics ---
    const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
    const totalSentimentScore = predictions.reduce((sum, p) => sum + parseInt(p.sentiment), 0);

    predictions.forEach(p => {
      sentimentCounts[p.sentiment]++;
    });

    const totalComments = comments.length;
    const totalWords = comments.reduce(
      (sum, c) => sum + c.split(/\s+/).filter(word => word.length > 0).length,
      0
    );
    const avgWordLength = (totalWords / totalComments).toFixed(2);
    const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
    const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

    // --- Summary metrics ---
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">Comment Analysis Summary</div>
        <div class="metrics-container">
          <div class="metric">
            <div class="metric-title">Total Comments</div>
            <div class="metric-value">${totalComments}</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg Comment Length</div>
            <div class="metric-value">${avgWordLength} words</div>
          </div>
          <div class="metric">
            <div class="metric-title">Avg Sentiment Score</div>
            <div class="metric-value">${normalizedSentimentScore}/10</div>
          </div>
        </div>
      </div>
    `;

    // --- Pie Chart Section ---
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">Sentiment Analysis Results</div>
        <p>See the pie chart below for sentiment distribution.</p>
        <div id="chart-container"></div>
      </div>`;
    await fetchAndDisplayChart(sentimentCounts);

    // --- Word Cloud Section ---
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">Comment Wordcloud</div>
        <div id="wordcloud-container"></div>
      </div>`;
    await fetchAndDisplayWordCloud(comments);

    // --- Top Comments ---
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">Top 25 Comments with Sentiments</div>
        <ul class="comment-list">
          ${predictions.slice(0, 25).map((item, i) => `
            <li class="comment-item">
              <span>${i + 1}. ${item.comment}</span><br>
              <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
            </li>`).join('')}
        </ul>
      </div>`;
  });

  // Fetch top-level comments only
  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`
        );
        const data = await response.json();

        if (data.items) {
          const newComments = data.items.map(
            item => item.snippet.topLevelComment.snippet.textOriginal
          );
          comments.push(...newComments);
        }
        if (!data.nextPageToken) break;
        pageToken = data.nextPageToken;
        await new Promise(r => setTimeout(r, 200)); // avoid rate limit
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += "<p>Error fetching comments.</p>";
    }
    console.log(`✅ Total comments fetched: ${comments.length}`);
    return comments;
  }

  // Predict sentiments via Flask backend
  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) return result;
      throw new Error(result.error || "Error fetching predictions");
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
      return null;
    }
  }

  // Pie chart
  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) throw new Error("Failed to fetch chart image");
      const blob = await response.blob();
      const img = document.createElement("img");
      img.src = URL.createObjectURL(blob);
      img.style.width = "100%";
      img.style.marginTop = "20px";
      document.getElementById("chart-container").appendChild(img);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      outputDiv.innerHTML += "<p>Error fetching chart image.</p>";
    }
  }

  // Wordcloud
  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) throw new Error("Failed to fetch word cloud image");
      const blob = await response.blob();
      const img = document.createElement("img");
      img.src = URL.createObjectURL(blob);
      img.style.width = "100%";
      img.style.marginTop = "20px";
      document.getElementById("wordcloud-container").appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      outputDiv.innerHTML += "<p>Error fetching word cloud image.</p>";
    }
  }
});
