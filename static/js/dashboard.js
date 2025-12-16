document.addEventListener('DOMContentLoaded', function() {
    
    // --- 1. ELEMENTS & VARIABLES ---
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImg');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsPanel = document.getElementById('resultsPanel');
    const emptyState = document.getElementById('emptyState');

    // --- 2. FILE UPLOAD HANDLING ---
    
    // Click to Open
    dropArea.addEventListener('click', () => fileInput.click());

    // File Selected
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) showPreview(file);
    });

    // Drag & Drop Effects
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Visual feedback on drag
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            uploadPlaceholder.classList.add('bg-indigo-50', 'border-indigo-400');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            uploadPlaceholder.classList.remove('bg-indigo-50', 'border-indigo-400');
        });
    });

    // Handle Drop
    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
        showPreview(files[0]);
    });

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImg.src = e.target.result;
            previewImg.classList.remove('hidden');
            previewImg.style.display = 'block'; // Force show
            uploadPlaceholder.classList.add('hidden');
            uploadPlaceholder.style.display = 'none'; // Force hide
        }
        reader.readAsDataURL(file);
    }

    // --- 3. FORM SUBMISSION (AJAX) ---
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!fileInput.files[0]) {
            Swal.fire({ icon: 'warning', title: 'No Scan', text: 'Please upload an X-Ray image.' });
            return;
        }

        // UI Loading State
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading loading-spinner"></span> Analyzing...';

        const formData = new FormData(this);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            // Success Alert
            Swal.fire({
                icon: 'success',
                title: 'Diagnosis Complete',
                text: 'AI analysis finished successfully.',
                timer: 1500,
                showConfirmButton: false
            });

            updateDashboard(data);

        } catch (error) {
            console.error(error);
            Swal.fire({ icon: 'error', title: 'Error', text: 'Analysis failed. See console.' });
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Scan';
        }
    });

    // --- 4. UPDATE DASHBOARD UI ---
    function updateDashboard(data) {
        // Hide Empty State, Show Results
        if(emptyState) emptyState.style.display = 'none';
        
        resultsPanel.classList.remove('hidden');
        resultsPanel.style.display = 'block';
        resultsPanel.scrollIntoView({ behavior: 'smooth' });

        // Update Text
        document.getElementById('resPrediction').innerText = data.prediction;
        
        // Dynamic Color for Result
        const resText = document.getElementById('resPrediction');
        if(data.prediction === 'Normal') {
            resText.classList.remove('text-red-600');
            resText.classList.add('text-green-600');
        } else {
            resText.classList.remove('text-green-600');
            resText.classList.add('text-red-600');
        }

        // Update Download Link
        document.getElementById('downloadBtn').href = `/download_report/${data.record_id}`;

        // Render Gauge Chart
        renderGauge(data.confidence);

        // Init Comparison Slider
        initSlider(data.original_url, data.overlay_url);
    }

    // --- 5. GAUGE CHART (UPDATED FOR LIGHT THEME) ---
    function renderGauge(score) {
        document.getElementById('confChart').innerHTML = ''; // Clear old chart
        
        const options = {
            series: [score],
            chart: { height: 220, type: 'radialBar' },
            plotOptions: {
                radialBar: {
                    hollow: { size: '65%' },
                    track: { background: "#f1f5f9" }, // Light grey track
                    dataLabels: {
                        show: true,
                        name: { 
                            show: true, 
                            color: '#64748b', // Slate 500 (Grey)
                            offsetY: -10
                        },
                        value: { 
                            show: true, 
                            fontSize: '26px', 
                            fontWeight: 'bold',
                            color: '#1e293b', // Slate 800 (Dark Blue/Black) - VISIBLE ON WHITE
                            offsetY: 5,
                            formatter: function(val) { return val + "%"; } 
                        }
                    }
                }
            },
            labels: ['Confidence'],
            colors: ['#4F46E5'], // Indigo color
            stroke: { lineCap: 'round' }
        };
        
        new ApexCharts(document.querySelector("#confChart"), options).render();
    }

    // --- 6. SLIDER LOGIC ---
    function initSlider(original, overlay) {
        const container = document.getElementById('compareContainer');
        const ts = new Date().getTime(); // Timestamp to prevent caching
        
        // Reset container
        container.innerHTML = '';
        container.classList.remove('hidden');
        document.getElementById('sliderLoading').classList.add('hidden');

        // Insert HTML structure
        container.innerHTML = `
            <div class="absolute inset-0 w-full h-full select-none">
                <img src="${overlay}?t=${ts}" class="w-full h-full object-contain block select-none" draggable="false">
            </div>
            <div id="imgOverlay" class="absolute inset-0 w-1/2 h-full overflow-hidden select-none border-r-2 border-white bg-slate-900">
                <img src="${original}?t=${ts}" class="w-[200%] max-w-none h-full object-contain block select-none" draggable="false" style="width: calc(100% / 0.5);">
            </div>
            <div id="sliderHandle" class="absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-1/2 w-10 h-10 bg-white rounded-full flex items-center justify-center cursor-ew-resize shadow-xl z-10 border border-indigo-100 hover:scale-110 transition-transform text-indigo-600">
                <i class="fa-solid fa-arrows-left-right"></i>
            </div>
        `;

        const sliderHandle = document.getElementById('sliderHandle');
        const imgOverlay = document.getElementById('imgOverlay');
        const imgInner = imgOverlay.querySelector('img');
        let isDown = false;

        // Interaction Logic
        const updateSlider = (x) => {
            const rect = container.getBoundingClientRect();
            let pos = ((x - rect.left) / rect.width) * 100;
            
            // Bounds
            if (pos < 0) pos = 0;
            if (pos > 100) pos = 100;
            
            sliderHandle.style.left = `${pos}%`;
            imgOverlay.style.width = `${pos}%`;
            // Counter-scaling to keep image stationary while container shrinks
            imgInner.style.width = `${10000/pos}%`; 
        };

        // Mouse Events
        sliderHandle.addEventListener('mousedown', () => isDown = true);
        window.addEventListener('mouseup', () => isDown = false);
        container.addEventListener('mousemove', (e) => { if (isDown) updateSlider(e.clientX); });

        // Touch Events (Mobile)
        sliderHandle.addEventListener('touchstart', () => isDown = true);
        window.addEventListener('touchend', () => isDown = false);
        container.addEventListener('touchmove', (e) => { if (isDown) updateSlider(e.touches[0].clientX); });
    }
});