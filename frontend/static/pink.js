/* =========================
   SPACE CAPTION GENERATOR
   PINK ALGORITHM TEAM
========================= */

// VLM Elements
const form = document.getElementById("aiForm");
const textInput = document.getElementById("textInput");
const imageInput = document.getElementById("imageInput");
const searchBtn = document.getElementById("searchBtn");
const resultBox = document.getElementById("vlmResultBox");

// Tab Elements
const tabVLM = document.getElementById("tabVLM");
const tabNEO = document.getElementById("tabNEO");
const tabCaption = document.getElementById("tabCaption");

const vlmTabContent = document.getElementById("vlmTabContent");
const neoTabContent = document.getElementById("neoTabContent");
const captionTabContent = document.getElementById("captionTabContent");

// NEO Elements
const neoForm = document.getElementById("neoForm");
const neoResultBox = document.getElementById("neoResultBox");

// Caption Elements
const captionForm = document.getElementById("captionForm");
const captionImageInput = document.getElementById("captionImageInput");
const generateCaptionBtn = document.getElementById("generateCaptionBtn");
const captionResultBox = document.getElementById("captionResultBox");

// Modal elements
const modal = document.getElementById("imageModal");
const modalImg = document.getElementById("modalImage");
const captionText = document.getElementById("captionText");
const modalClose = document.getElementById("modalClose");

// Set the base URL for the FastAPI backend running on port 8000
const FASTAPI_URL = "http://127.0.0.1:8000";

/* =========================
   TAB SWITCHING LOGIC
========================= */
const tabs = ["VLM", "Caption", "NEO"];

function switchTab(activeTab) {
  tabs.forEach((tab) => {
    const btn = document.getElementById("tab" + tab);
    const content = document.getElementById(tab.toLowerCase() + "TabContent");
    if (btn && content) {
      if (tab === activeTab) {
        btn.classList.add("active");
        content.classList.add("active");
      } else {
        btn.classList.remove("active");
        content.classList.remove("active");
      }
    }
  });
}

if (tabVLM) {
  tabVLM.addEventListener("click", () => switchTab("VLM"));
  tabNEO.addEventListener("click", () => switchTab("NEO"));
  tabCaption.addEventListener("click", () => switchTab("Caption"));
}

/* =========================
   NEO SLIDERS REAL-TIME UPDATE
========================= */
const slidersConfig = [
  { id: "neo_h", suffix: "" },
  { id: "neo_dia_min", suffix: " km", isDecimal: true },
  { id: "neo_dia_max", suffix: " km", isDecimal: true },
  { id: "neo_vel", suffix: " km/h", isLocale: true },
  { id: "neo_miss", suffix: " km", isLocale: true },
];

slidersConfig.forEach((cfg) => {
  const slider = document.getElementById(cfg.id);
  const valDisplay = document.getElementById(cfg.id + "_val");

  if (slider && valDisplay) {
    slider.addEventListener("input", () => {
      let formattedVal = slider.value;
      if (cfg.isDecimal) {
        formattedVal = Number(slider.value).toFixed(3);
      } else if (cfg.isLocale) {
        formattedVal = Number(slider.value).toLocaleString("en-US");
      }
      valDisplay.textContent = formattedVal + cfg.suffix;
    });
  }
});

/* =========================
   VLM INPUT VALIDATION
========================= */
function validateInputs() {
  const hasText = textInput.value.trim().length > 0;
  const hasImage = imageInput.files.length > 0;

  // Only allow if ONE is filled out, not both.
  if ((hasText && !hasImage) || (!hasText && hasImage)) {
    searchBtn.disabled = false;
  } else {
    searchBtn.disabled = true;
  }

  // Visual cues (disable the other input)
  if (hasText) {
    imageInput.disabled = true;
  } else {
    imageInput.disabled = false;
  }

  if (hasImage) {
    textInput.disabled = true;
  } else {
    textInput.disabled = false;
  }
}

if (textInput && imageInput) {
  textInput.addEventListener("input", validateInputs);
  imageInput.addEventListener("change", validateInputs);
}

/* =========================
   FORM SUBMIT (SEARCH)
========================= */
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const hasText = textInput.value.trim().length > 0;
    const hasImage = imageInput.files.length > 0;

    resultBox.innerHTML = `
      <div style="text-align: center;">
          <p style="color:#00d4ff;">Searching cosmic database... 🚀</p>
          <div class="loader"></div>
      </div>
    `;

    try {
      let endpoint = "";
      const formData = new FormData();

      const vlmModelSelect = document.getElementById("vlm_model_select");
      if (vlmModelSelect) {
        formData.append("model_id", vlmModelSelect.value);
      }

      if (hasText) {
        endpoint = `${FASTAPI_URL}/api/siglip/search_text`;
        formData.append("query", textInput.value.trim());
      } else if (hasImage) {
        endpoint = `${FASTAPI_URL}/api/siglip/search_image`;
        formData.append("image", imageInput.files[0]);
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        resultBox.innerHTML = `<p style="color:red;">Error: ${data.detail} ❌</p>`;
        return;
      }

      let captionHtml = "";
      if (data.caption) {
        captionHtml = `<p style="color:#aaa; font-style:italic;"><strong>Query Image Caption:</strong> ${data.caption}</p>`;
      }

      resultBox.innerHTML = `
        <h3 style="color:#ff6ec7;">Search Results 🌌</h3>
        ${captionHtml}
        <p style="color: #7b61ff; margin-bottom: 10px;">Click an image to enlarge and view its info:</p>
        <div style="display: flex; gap: 15px; flex-wrap: wrap; justify-content: center;">
            ${data.results
              .map(
                (r, index) => `
                <img src="${r.url}" 
                     class="gallery-image"
                     data-caption="${r.caption.replace(/"/g, "&quot;")}"
                     data-title="${r.title.replace(/"/g, "&quot;")}"
                     style="width: 150px; height: 150px; object-fit: cover; border-radius: 12px; border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 4px 15px rgba(0,0,0,0.5); cursor: pointer; transition: transform 0.2s;">
            `,
              )
              .join("")}
        </div>
      `;

      // Bind modal click events
      document.querySelectorAll(".gallery-image").forEach((img) => {
        img.addEventListener("mouseover", () => {
          img.style.transform = "scale(1.05)";
        });
        img.addEventListener("mouseout", () => {
          img.style.transform = "scale(1)";
        });

        img.addEventListener("click", () => {
          modal.style.display = "block";
          modalImg.src = img.src;
          captionText.innerHTML = `<h3 style="color:#ff6ec7;">${img.getAttribute("data-title")}</h3><p>${img.getAttribute("data-caption")}</p>`;
        });
      });
    } catch (error) {
      resultBox.innerHTML = `<p style="color:red;">Backend connection error ❌ (Make sure FastAPI is running on port 8000)</p>`;
    }
  });
}

/* =========================
   FORM SUBMIT (CAPTION GENERATOR)
========================= */
if (captionForm) {
  captionForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!captionImageInput.files.length) return;

    captionResultBox.style.display = "block";
    captionResultBox.innerHTML = `
      <div style="text-align: center;">
          <p style="color:#00d4ff;">ViT-GPT2 is analyzing image... 🤖</p>
          <div class="loader"></div>
      </div>
    `;

    try {
      const formData = new FormData();
      formData.append("image", captionImageInput.files[0]);

      const captionModelSelect = document.getElementById("caption_model_select");
      if (captionModelSelect) {
        formData.append("model_id", captionModelSelect.value);
      }

      const response = await fetch(`${FASTAPI_URL}/api/caption/generate`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        captionResultBox.innerHTML = `<p style="color:red;">Error: ${data.detail} ❌</p>`;
        return;
      }

      captionResultBox.innerHTML = `
        <h3 style="color:#ff6ec7; text-align: center;">AI Generated Description</h3>
        <p style="text-align: center; color: white; font-size: 18px; margin-top: 15px; border-left: 4px solid #7b61ff; padding-left: 15px; background: rgba(123, 97, 255, 0.1); border-radius: 5px; padding: 15px; display: inline-block; width: 100%;">
          "${data.caption}"
        </p>
      `;
    } catch (error) {
      captionResultBox.innerHTML = `<p style="color:red;">Backend connection error ❌</p>`;
    }
  });
}

/* =========================
   FORM SUBMIT (NEO PREDICTOR)
========================= */
if (neoForm) {
  neoForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    neoResultBox.style.display = "block";
    neoResultBox.innerHTML = `
      <div style="text-align: center;">
          <p style="color:#ff6ec7;">Running Random Forest Simulation... 🚀</p>
          <div class="loader"></div>
      </div>
    `;

    try {
      const formData = new FormData();
      formData.append("h", document.getElementById("neo_h").value);
      formData.append("dia_min", document.getElementById("neo_dia_min").value);
      formData.append("dia_max", document.getElementById("neo_dia_max").value);
      formData.append("vel", document.getElementById("neo_vel").value);
      formData.append("miss", document.getElementById("neo_miss").value);

      const response = await fetch(`${FASTAPI_URL}/api/neo/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        neoResultBox.innerHTML = `<p style="color:red;">Error: ${data.detail} ❌</p>`;
        return;
      }

      // Convert Markdown-like text to basic HTML tags
      const htmlExplanation = data.explanation
        .replace(/\n/g, "<br>")
        .replace(/## /g, "<strong>")
        .replace(/### /g, "<strong><em>")
        .replace(/---/g, "<hr>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>");

      const isSafe = data.label.includes("NOT");
      const labelColor = isSafe ? "#00ffcc" : "#ff3333";

      neoResultBox.innerHTML = `
        <h3 style="color:${labelColor}; text-align:center;">${data.label}</h3>
        <p style="text-align:center;"><strong>AI Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%</p>
        
        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin: 20px 0;">
        
        <div style="font-size:14px; text-align: left; background: rgba(0,0,0,0.4); padding: 20px; border-radius: 12px; border-left: 4px solid ${labelColor}; line-height: 1.8;">
            ${htmlExplanation}
        </div>
      `;
    } catch (error) {
      neoResultBox.innerHTML = `<p style="color:red;">Backend connection error ❌ (Make sure FastAPI is running on port 8000)</p>`;
    }
  });
}

/* =========================
   MODAL LOGIC
========================= */
if (modalClose) {
  modalClose.onclick = () => {
    modal.style.display = "none";
  };
}

// Click outside modal to close
window.onclick = (event) => {
  if (event.target === modal) {
    modal.style.display = "none";
  }
};

/* =========================
   BUTTON CLICK EFFECT
========================= */
document.querySelectorAll(".btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (!btn.disabled) {
      btn.style.transform = "scale(0.95)";
      setTimeout(() => {
        btn.style.transform = "scale(1)";
      }, 100);
    }
  });
});
