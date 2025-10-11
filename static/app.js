// Enhanced loading and error handling
function showLoadingState(message = 'Loading...') {
  const filterStatus = document.getElementById('filter-status');
  if (filterStatus) {
    filterStatus.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <div class="spinner"></div>
        <span>${message}</span>
      </div>
    `;
    filterStatus.style.color = 'var(--primary)';
  }
}

function showErrorState(message) {
  const filterStatus = document.getElementById('filter-status');
  if (filterStatus) {
    filterStatus.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px; color: #e74c3c;">
        <span style="font-size: 18px;">⚠️</span>
        <span>${message}</span>
      </div>
    `;
    filterStatus.style.color = '#e74c3c';
  }
}

function showSuccessState(message) {
  const filterStatus = document.getElementById('filter-status');
  if (filterStatus) {
    filterStatus.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px; color: #27ae60;">
        <span style="font-size: 18px;">✅</span>
        <span>${message}</span>
      </div>
    `;
    filterStatus.style.color = '#27ae60';
  }
}

// Add CSS for spinner
const spinnerCSS = `
  .spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

// Inject spinner CSS
const style = document.createElement('style');
style.textContent = spinnerCSS;
document.head.appendChild(style);

// Debouncing utility to reduce redundant API calls
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Chart update debounced to prevent excessive API calls
const debouncedUpdateCharts = debounce(updateCharts, 300);

// New insightful chart rendering functions
function renderInsightfulCharts(chartsData) {
    console.log('🎨 Rendering insightful charts:', Object.keys(chartsData));
    
    // Clear existing charts
    clearAllCharts();
    
    // Render each chart type
    Object.entries(chartsData).forEach(([chartKey, chartData]) => {
        try {
            switch (chartData.type) {
                case 'radar':
                    renderRadarChart(chartKey, chartData);
                    break;
                case 'heatmap':
                    renderHeatmapChart(chartKey, chartData);
                    break;
                case 'multi_line':
                    renderMultiLineChart(chartKey, chartData);
                    break;
                case 'network':
                    renderNetworkChart(chartKey, chartData);
                    break;
                case 'box_plot':
                    renderBoxPlotChart(chartKey, chartData);
                    break;
                case 'pie_with_insights':
                    renderPieWithInsightsChart(chartKey, chartData);
                    break;
                case 'scatter':
                    renderScatterChart(chartKey, chartData);
                    break;
                case 'comparison':
                    renderComparisonChart(chartKey, chartData);
                    break;
                case 'bar':
                    renderBarChart(chartKey, chartData);
                    break;
                default:
                    console.warn(`Unknown chart type: ${chartData.type}`);
            }
        } catch (error) {
            console.error(`Error rendering ${chartKey}:`, error);
        }
    });
}

function renderRadarChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    const datasets = data.genres.map((genre, index) => ({
        label: genre,
        data: data.features.map(feature => data.profiles[genre][feature]),
        borderColor: getChartColor(index),
        backgroundColor: getChartColor(index, 0.2),
        borderWidth: 2,
        pointBackgroundColor: getChartColor(index),
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: getChartColor(index)
    }));
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.features,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                        font: {
                            size: 12
                        }
                    },
                    pointLabels: {
                        font: {
                            size: 14
                        }
                    }
                }
            }
        }
    });
}

function renderHeatmapChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    // Create a proper heatmap using Chart.js instead of manual drawing
    const datasets = [{
        label: 'Genre Preference %',
        data: [],
        backgroundColor: [],
        borderColor: [],
        borderWidth: 1
    }];
    
    // Convert matrix data to Chart.js format
    data.matrix.forEach((row, rowIndex) => {
        row.forEach((value, colIndex) => {
            datasets[0].data.push({
                x: colIndex,
                y: rowIndex,
                v: value
            });
        });
    });
    
    // Create a bubble chart as heatmap alternative
    new Chart(ctx, {
        type: 'bubble',
        data: {
            datasets: [{
                label: 'Genre Preference',
                data: datasets[0].data.map(point => ({
                    x: point.x,
                    y: point.y,
                    r: Math.max(5, Math.min(20, point.v * 0.5)) // Size based on value
                })),
                backgroundColor: datasets[0].data.map(point => {
                    const intensity = point.v / Math.max(...data.matrix.flat());
                    return `rgba(54, 162, 235, ${intensity})`;
                }),
                borderColor: '#fff',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            const country = data.countries[point.y];
                            const genre = data.genres[point.x];
                            const value = data.matrix[point.y][point.x];
                            return `${country} - ${genre}: ${value.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: -0.5,
                    max: data.genres.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            return data.genres[Math.round(value)] || '';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Genres'
                    }
                },
                y: {
                    min: -0.5,
                    max: data.countries.length - 0.5,
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            return data.countries[Math.round(value)] || '';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Countries'
                    }
                }
            }
        }
    });
}

function renderMultiLineChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    const datasets = [
        {
            label: 'Yearly Trends',
            data: Object.values(data.yearly),
            borderColor: '#e74c3c',
            backgroundColor: '#e74c3c',
            tension: 0.4
        },
        {
            label: 'Monthly Seasonality',
            data: Object.values(data.monthly),
            borderColor: '#3498db',
            backgroundColor: '#3498db',
            tension: 0.4
        },
        {
            label: 'Weekly Patterns',
            data: Object.values(data.weekly),
            borderColor: '#2ecc71',
            backgroundColor: '#2ecc71',
            tension: 0.4
        }
    ];
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: Object.keys(data.yearly),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Releases'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time Period'
                    }
                }
            }
        }
    });
}

function renderBoxPlotChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    // Create proper box plot data with statistical values
    const datasets = [{
        label: 'Popularity Distribution',
        data: data.genres.map(genre => {
            const stats = data.data[genre];
            return {
                min: stats.min,
                q1: stats.q1,
                median: stats.median,
                q3: stats.q3,
                max: stats.max,
                mean: stats.mean
            };
        }),
        backgroundColor: data.genres.map((_, index) => getChartColor(index, 0.3)),
        borderColor: data.genres.map((_, index) => getChartColor(index)),
        borderWidth: 2
    }];
    
    // Use bar chart to show mean values with error bars
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.genres,
            datasets: [{
                label: 'Average Popularity',
                data: data.genres.map(genre => data.data[genre].mean),
                backgroundColor: data.genres.map((_, index) => getChartColor(index, 0.6)),
                borderColor: data.genres.map((_, index) => getChartColor(index)),
                borderWidth: 2,
                errorBars: data.genres.map(genre => ({
                    plus: data.data[genre].std,
                    minus: data.data[genre].std
                }))
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const genre = context.label;
                            const stats = data.data[genre];
                            return [
                                `Mean: ${stats.mean}`,
                                `Median: ${stats.median}`,
                                `Min: ${stats.min}, Max: ${stats.max}`,
                                `Std Dev: ${stats.std}`
                            ];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Popularity Score'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Music Genres'
                    }
                }
            }
        }
    });
}

function renderPieWithInsightsChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    const labels = Object.keys(data.data);
    const values = Object.values(data.data).map(item => item.percentage);
    const colors = labels.map((_, index) => getChartColor(index));
    
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'right',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12
                        },
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const genre = context.label;
                            const item = data.data[genre];
                            return `${genre}: ${item.percentage}% (${item.count} songs)`;
                        }
                    }
                }
            }
        }
    });
}

function renderScatterChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    // Reduce data density by sampling and clustering
    const datasets = Object.entries(data.data).map(([genre, genreData], index) => {
        // Sample data to reduce density (take every 3rd point)
        const sampledData = [];
        for (let i = 0; i < genreData.energy.length; i += 3) {
            sampledData.push({
                x: genreData.energy[i],
                y: genreData.valence[i],
                popularity: genreData.popularity[i]
            });
        }
        
        return {
            label: genre,
            data: sampledData,
            backgroundColor: getChartColor(index, 0.4),
            borderColor: getChartColor(index),
            pointRadius: 3,
            pointHoverRadius: 6,
            pointBorderWidth: 1
        };
    });
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return `${context.dataset.label}: Energy=${point.x.toFixed(2)}, Valence=${point.y.toFixed(2)}, Popularity=${point.popularity}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Energy'
                    },
                    min: 0,
                    max: 1
                },
                y: {
                    title: {
                        display: true,
                        text: 'Valence'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

function renderComparisonChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Explicit', 'Clean'],
            datasets: [{
                label: 'Song Count',
                data: [data.explicit.count, data.clean.count],
                backgroundColor: ['#e74c3c', '#2ecc71'],
                borderColor: ['#c0392b', '#27ae60'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const isExplicit = context.label === 'Explicit';
                            const item = isExplicit ? data.explicit : data.clean;
                            return `Average: ${item.mean}, Count: ${item.count} (${item.percentage}%)`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Songs'
                    },
                    ticks: {
                        callback: function(value) {
                            return value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

function renderBarChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    // Determine if it's countries or artists data
    const labels = data.countries || data.artists || [];
    const values = data.song_counts || [];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: data.countries ? 'Song Count' : 'Song Count',
                data: values,
                backgroundColor: values.map((_, index) => getChartColor(index, 0.6)),
                borderColor: values.map((_, index) => getChartColor(index)),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            resizeDelay: 0,
            plugins: {
                title: {
                    display: true,
                    text: data.title,
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed.y.toLocaleString()} songs`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Songs'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: data.countries ? 'Countries' : 'Artists'
                    }
                }
            }
        }
    });
}

function renderNetworkChart(chartKey, data) {
    const canvas = createChartCanvas(chartKey, data.title, data.insight);
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions properly
    canvas.width = 800;
    canvas.height = 500;
    
    // High-resolution circular network visualization
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) / 2.5; // Smaller radius
    
    // Draw nodes
    data.features.forEach((feature, index) => {
        const angle = (2 * Math.PI * index) / data.features.length;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        // Draw smaller node
        ctx.beginPath();
        ctx.arc(x, y, 20, 0, 2 * Math.PI); // Smaller radius
        ctx.fillStyle = getChartColor(index);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw label
        ctx.fillStyle = '#000';
        ctx.font = 'bold 13px Arial'; // Slightly smaller, bold font
        ctx.textAlign = 'center';
        ctx.fillText(feature, x, y + 3);
    });
    
    // Draw connections
    data.correlations.forEach(corr => {
        const feature1Index = data.features.indexOf(corr.feature1);
        const feature2Index = data.features.indexOf(corr.feature2);
        
        const angle1 = (2 * Math.PI * feature1Index) / data.features.length;
        const angle2 = (2 * Math.PI * feature2Index) / data.features.length;
        
        const x1 = centerX + radius * Math.cos(angle1);
        const y1 = centerY + radius * Math.sin(angle1);
        const x2 = centerX + radius * Math.cos(angle2);
        const y2 = centerY + radius * Math.sin(angle2);
        
        // High-resolution line width based on correlation strength
        const lineWidth = Math.max(1, Math.abs(corr.correlation) * 3); // Thinner, cleaner lines
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = corr.correlation > 0 ? '#2ecc71' : '#e74c3c';
        ctx.globalAlpha = Math.max(0.5, Math.abs(corr.correlation));
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        
        ctx.globalAlpha = 1;
    });
}

// Helper functions
function createChartCanvas(chartKey, title, insight) {
    const container = document.getElementById('charts-container');
    if (!container) {
        console.error('Charts container not found');
        return null;
    }
    
    // Create chart wrapper
    const chartWrapper = document.createElement('div');
    chartWrapper.className = 'chart-wrapper';
    chartWrapper.style.cssText = `
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        max-width: 100%;
        overflow: visible;
        height: auto;
        min-height: 600px;
        max-height: none;
    `;
    
    // Create title
    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    titleEl.style.cssText = `
        margin: 0 0 10px 0;
        color: #2c3e50;
        font-size: 18px;
        font-weight: bold;
    `;
    
    // Create insight
    const insightEl = document.createElement('p');
    insightEl.textContent = insight;
    insightEl.style.cssText = `
        margin: 0 0 15px 0;
        color: #7f8c8d;
        font-size: 14px;
        font-style: italic;
    `;
    
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.id = `chart-${chartKey}`;
    canvas.style.cssText = `
        width: 100%;
        height: 500px;
        max-width: 100%;
        max-height: 500px;
        object-fit: contain;
    `;
    
    chartWrapper.appendChild(titleEl);
    chartWrapper.appendChild(insightEl);
    chartWrapper.appendChild(canvas);
    container.appendChild(chartWrapper);
    
    return canvas;
}

function getChartColor(index, alpha = 1) {
    const colors = [
        `rgba(54, 162, 235, ${alpha})`,   // Blue
        `rgba(255, 99, 132, ${alpha})`,  // Red
        `rgba(255, 205, 86, ${alpha})`,   // Yellow
        `rgba(75, 192, 192, ${alpha})`,  // Teal
        `rgba(153, 102, 255, ${alpha})`,  // Purple
        `rgba(255, 159, 64, ${alpha})`,   // Orange
        `rgba(199, 199, 199, ${alpha})`,  // Gray
        `rgba(83, 102, 255, ${alpha})`,   // Indigo
        `rgba(255, 99, 255, ${alpha})`,   // Pink
        `rgba(99, 255, 132, ${alpha})`    // Green
    ];
    return colors[index % colors.length];
}

function clearAllCharts() {
    const container = document.getElementById('charts-container');
    if (container) {
        container.innerHTML = '';
    }
}

// Cache for API responses to avoid redundant calls
const apiCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function getCachedData(key) {
  const cached = apiCache.get(key);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data;
  }
  return null;
}

function setCachedData(key, data) {
  apiCache.set(key, {
    data: data,
    timestamp: Date.now()
  });
}

let chartInstances = {}; // Store chart instances for updates
let container = null; // Global container reference

// Optimized fetch with caching
async function fetchWithCache(url, options = {}) {
  const cacheKey = `${url}${JSON.stringify(options)}`;
  const cached = getCachedData(cacheKey);
  
  if (cached) {
    console.log('📦 Using cached data for:', url);
    return {
      json: () => Promise.resolve(cached)
    };
  }
  
  console.log('🌐 Fetching fresh data from:', url);
  const response = await fetch(url, options);
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const data = await response.json();
  setCachedData(cacheKey, data);
  
  return {
    json: () => Promise.resolve(data)
  };
}

// Dynamic filtering functions
function getCurrentFilters() {
  const filters = new URLSearchParams();
  
  // Get selected countries
  const countrySelect = document.getElementById('country');
  if (countrySelect) {
    const selectedCountries = Array.from(countrySelect.selectedOptions)
      .map(option => option.value)
      .filter(v => v);
    selectedCountries.forEach(country => filters.append('country', country));
  }
  
  // Get other filters
  const genre = document.getElementById('genre')?.value;
  const subgenre = document.getElementById('subgenre')?.value;
  const language = document.getElementById('language')?.value;
  const popMin = document.getElementById('popMin')?.value;
  const popMax = document.getElementById('popMax')?.value;
  const releaseStart = document.getElementById('releaseStart')?.value;
  const releaseEnd = document.getElementById('releaseEnd')?.value;
  const trend = document.getElementById('trend')?.value;
  
  if (genre) filters.append('genre', genre);
  if (subgenre) filters.append('subgenre', subgenre);
  if (language) filters.append('language', language);
  if (popMin) filters.append('popMin', popMin);
  if (popMax) filters.append('popMax', popMax);
  if (releaseStart) filters.append('releaseStart', releaseStart);
  if (releaseEnd) filters.append('releaseEnd', releaseEnd);
  if (trend) filters.append('trend', trend);
  
  return filters;
}

async function loadFilterOptions() {
  try {
    console.log('🔄 Loading filter options...');
    showLoadingState('Loading filter options...');
    
    const response = await fetchWithCache('/api/filter-options');
    const result = await response.json();
    
    if (result.status === 'error') {
      showErrorState(result.message || 'Error loading filter options');
      return;
    }
    
    const options = result.data;
    console.log('✅ Filter options loaded:', options);
    
    // Populate country dropdown
    const countrySelect = document.getElementById('country');
    if (countrySelect && options.countries) {
      countrySelect.innerHTML = '<option value="">All Countries</option>';
      options.countries.forEach(country => {
        const option = document.createElement('option');
        option.value = country.code;
        option.textContent = country.name;
        countrySelect.appendChild(option);
      });
    }
    
    // Populate genre dropdown
    const genreSelect = document.getElementById('genre');
    if (genreSelect && options.genres) {
      genreSelect.innerHTML = '<option value="">All Genres</option>';
      options.genres.forEach(genre => {
        const option = document.createElement('option');
        option.value = genre;
        option.textContent = genre;
        genreSelect.appendChild(option);
      });
    }
    
    // Populate other dropdowns similarly...
    showSuccessState('Filter options loaded successfully!');
    
  } catch (error) {
    console.error('Failed to load filter options:', error);
    showErrorState(`Failed to load filters: ${error.message}`);
  }
}

async function updateCharts() {
  try {
    console.log('Starting insightful chart update...');
    showLoadingState('Generating insightful analytics...');
    
    // Get current filters and fetch insightful chart data
    const filters = getCurrentFilters();
    const url = `/api/chart-data?${new URLSearchParams(filters).toString()}`;
    
    console.log('🔄 Fetching insightful charts from:', url);
    const response = await fetchWithCache(url);
    const result = await response.json();
    
    if (result.status === 'error') {
      showErrorState(result.message || 'Error loading charts');
      return;
    }
    
    const chartsData = result.data;
    console.log('✅ Loaded insightful charts:', Object.keys(chartsData));
    
    // Render the new insightful charts
    renderInsightfulCharts(chartsData);
    
    // Show success message with insights
    const chartCount = Object.keys(chartsData).length;
    showSuccessState(`${chartCount} insightful analytics charts generated!`);
    
  } catch (error) {
    console.error('❌ Error updating charts:', error);
    showErrorState('Failed to load chart data');
  }
}

function setupEventListeners() {
  // Add event listeners to all filter elements
  const filterElements = [
    'country', 'genre', 'subgenre', 'language', 
    'popMin', 'popMax', 'releaseStart', 'releaseEnd', 'trend'
  ];
  
  filterElements.forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener('change', debouncedUpdateCharts);
    }
  });
  
  // Add event listener to Apply button
  const applyButton = document.querySelector('button[type="submit"]');
  if (applyButton) {
    applyButton.addEventListener('click', (e) => {
      e.preventDefault();
      updateCharts(); // Use immediate update for button clicks
    });
  }
  
  // Add event listener to Reset button
  const resetButton = document.getElementById('resetFilters');
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      // Reset all filters
      document.getElementById('country').selectedIndex = -1;
      document.getElementById('genre').value = '';
      document.getElementById('subgenre').value = '';
      document.getElementById('language').value = '';
      document.getElementById('popMin').value = document.getElementById('popMin').min;
      document.getElementById('popMax').value = document.getElementById('popMax').max;
      document.getElementById('releaseStart').value = '';
      document.getElementById('releaseEnd').value = '';
      document.getElementById('trend').value = '';
      
      // Clear cache when resetting
      apiCache.clear();
      
      // Update charts with no filters
      updateCharts();
    });
  }
}

document.addEventListener('DOMContentLoaded', async ()=>{
  console.log('🚀 DOM Content Loaded - Starting Spotify Analytics Dashboard');
  
  // Get the charts container
  container = document.getElementById('charts-container');
  if (!container) {
    console.error('❌ Charts container not found!');
    return;
  }
  console.log('✅ Charts container found');
  
  // DOM loaded, Chart.js available
  container.innerHTML = '';
  
  if (typeof Chart === 'undefined') {
    console.error("❌ Chart.js not loaded!");
    return;
  }
  
  console.log('✅ Chart.js loaded successfully');
  
  // Setup event listeners
  setupEventListeners();
  
  // Load initial data
  console.log('🔄 Loading initial data...');
  await loadFilterOptions();
  await updateCharts();
  console.log('✅ Initial data loading completed');
});