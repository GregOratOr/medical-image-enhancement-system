# app/components/dual_view.py

import streamlit.components.v1 as components

def render_dual_view(img1_b64: str, img2_b64: str, height: int = 500):
    """
    Renders two images side-by-side with synchronized pan and zoom.
    
    Args:
        img1_b64: Base64 string of the first image (Original).
        img2_b64: Base64 string of the second image (Denoised).
        height: Height of the viewport in pixels.
    """
    
    # Inject a full HTML/JS app into the Streamlit iframe
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background-color: #0e1117; color: white; font-family: sans-serif; }}
            .container {{
                display: flex;
                flex-direction: row;
                height: {height}px;
                width: 100%;
                gap: 10px;
            }}
            .viewport {{
                flex: 1;
                position: relative;
                overflow: hidden;
                border: 1px solid #333;
                border-radius: 4px;
                background-color: #000;
                cursor: grab;
            }}
            .viewport:active {{ cursor: grabbing; }}
            .label {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.7);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                pointer-events: none;
                z-index: 10;
            }}
            img {{
                position: absolute;
                transform-origin: 0 0;
                will-change: transform;
                /* Disable default drag behavior to allow our custom panning */
                user-select: none;
                -webkit-user-drag: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="viewport" id="view1">
                <div class="label">Original (Noisy)</div>
                <img id="img1" src="{img1_b64}" draggable="false">
            </div>
            <div class="viewport" id="view2">
                <div class="label">Denoised (Clean)</div>
                <img id="img2" src="{img2_b64}" draggable="false">
            </div>
        </div>

        <script>
            // --- STATE MANAGEMENT ---
            let state = {{
                scale: 1,
                panning: false,
                pointX: 0,
                pointY: 0,
                startX: 0,
                startY: 0
            }};

            const img1 = document.getElementById('img1');
            const img2 = document.getElementById('img2');
            const view1 = document.getElementById('view1');
            const view2 = document.getElementById('view2');

            // --- TRANSFORM FUNCTION ---
            function setTransform() {{
                const transform = `translate(${{state.pointX}}px, ${{state.pointY}}px) scale(${{state.scale}})`;
                img1.style.transform = transform;
                img2.style.transform = transform;
            }}

            // --- ZOOM LOGIC (Wheel) ---
            function handleZoom(e) {{
                e.preventDefault();
                
                const xs = (e.clientX - state.pointX) / state.scale;
                const ys = (e.clientY - state.pointY) / state.scale;
                
                const delta = -e.deltaY;
                
                // Zoom factor (1.1x per scroll tick)
                (delta > 0) ? (state.scale *= 1.1) : (state.scale /= 1.1);
                
                // Limit zoom (0.1x to 50x)
                state.scale = Math.min(Math.max(0.1, state.scale), 50);

                state.pointX = e.clientX - xs * state.scale;
                state.pointY = e.clientY - ys * state.scale;

                setTransform();
            }}

            // --- PAN LOGIC (Mouse Drag) ---
            function handleMouseDown(e) {{
                e.preventDefault();
                state.startX = e.clientX - state.pointX;
                state.startY = e.clientY - state.pointY;
                state.panning = true;
            }}

            function handleMouseUp(e) {{
                state.panning = false;
            }}

            function handleMouseMove(e) {{
                if (!state.panning) return;
                e.preventDefault();
                state.pointX = e.clientX - state.startX;
                state.pointY = e.clientY - state.startY;
                setTransform();
            }}

            // --- ATTACH LISTENERS ---
            // We attach to both viewports so you can control from either side
            [view1, view2].forEach(view => {{
                view.addEventListener('wheel', handleZoom);
                view.addEventListener('mousedown', handleMouseDown);
                window.addEventListener('mouseup', handleMouseUp);
                window.addEventListener('mousemove', handleMouseMove);
            }});
            
            // Initial render
            // Fit images to width roughly (assuming square images)
            const initialScale = Math.min(view1.offsetWidth / 512, view1.offsetHeight / 512); 
            // state.scale = initialScale; // Optional: auto-fit
            setTransform();

        </script>
    </body>
    </html>
    """
    
    # Render with Streamlit
    components.html(html_code, height=height)