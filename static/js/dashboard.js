document.addEventListener('DOMContentLoaded', function() {

    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImg');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const uploadForm = document.getElementById('uploadForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsPanel = document.getElementById('resultsPanel');
    const emptyState = document.getElementById('emptyState');

    dropArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function () {
        if (this.files[0]) showPreview(this.files[0]);
    });

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            previewImg.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        if (!fileInput.files[0]) {
            Swal.fire({ icon: 'warning', title: 'No Scan', text: 'Please upload an X-Ray image.' });
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = 'Analyzing...';

        const formData = new FormData(this);

        try {
            const response = await fetch('/analyze', { method: 'POST', body: formData });
            const data = await response.json();

            // 🔴 HANDLE ERROR RESPONSE
            if (data.status !== "success") {

                if (resultsPanel) resultsPanel.style.display = "none";
                if (emptyState) emptyState.style.display = "block";

                Swal.fire({
                    icon: 'warning',
                    title: data.title || 'AI Unavailable',
                    text: data.message,
                    confirmButtonText: 'OK'
                });

                return;
            }

            // ✅ SUCCESS (local / paid env only)
            Swal.fire({
                icon: 'success',
                title: 'Diagnosis Complete',
                timer: 1200,
                showConfirmButton: false
            });

            updateDashboard(data);

        } catch (err) {
            Swal.fire({ icon: 'error', title: 'Error', text: 'Server error occurred.' });
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = 'Analyze Scan';
        }
    });

    function updateDashboard(data) {
        if (!data || !data.prediction) return;

        if (emptyState) emptyState.style.display = "none";
        resultsPanel.style.display = "block";
        resultsPanel.scrollIntoView({ behavior: 'smooth' });

        document.getElementById('resPrediction').innerText = data.prediction;
    }
});
