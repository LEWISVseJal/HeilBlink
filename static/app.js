document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("js-ready");

  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  setupRevealAnimations(prefersReducedMotion);
  setupImagePreview();
  setupDashboardLoader(prefersReducedMotion);
  setupProgressAnimations(prefersReducedMotion);
  setupCountups(prefersReducedMotion);
});

function setupRevealAnimations(prefersReducedMotion) {
  const revealItems = document.querySelectorAll(".fade-up");

  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    revealItems.forEach((item) => item.classList.add("is-visible"));
    return;
  }

  if (!revealItems.length) {
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    {
      threshold: 0.14,
      rootMargin: "0px 0px -8% 0px",
    }
  );

  revealItems.forEach((item) => observer.observe(item));
}

function setupImagePreview() {
  const imageInput = document.querySelector("[data-image-input]");
  const previewImage = document.querySelector("[data-image-preview]");
  const previewPlaceholder = document.querySelector("[data-image-placeholder]");
  const fileName = document.querySelector("[data-file-name]");
  const uploadZone = imageInput ? imageInput.closest(".upload-dropzone") : null;

  if (!imageInput || !previewImage || !previewPlaceholder || !fileName) {
    return;
  }

  imageInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];

    if (!file) {
      uploadZone?.classList.remove("is-selected");
      previewImage.classList.add("is-hidden");
      previewImage.removeAttribute("src");
      previewPlaceholder.classList.remove("is-hidden");
      fileName.textContent = "No file selected";
      return;
    }

    uploadZone?.classList.add("is-selected");
    fileName.textContent = file.name;
    const reader = new FileReader();

    reader.onload = (loadEvent) => {
      previewImage.src = loadEvent.target.result;
      previewImage.classList.remove("is-hidden");
      previewPlaceholder.classList.add("is-hidden");
    };

    reader.readAsDataURL(file);
  });
}

function setupDashboardLoader(prefersReducedMotion) {
  const loader = document.querySelector("[data-model-loader]");

  if (!loader) {
    return;
  }

  const fill = loader.querySelector("[data-loader-fill]");
  const percent = loader.querySelector("[data-loader-percent]");
  const status = loader.querySelector("[data-loader-status]");

  if (!fill || !percent || !status) {
    return;
  }

  const steps = [
    { threshold: 0, label: "Inspecting comparison studies..." },
    { threshold: 35, label: "Ranking disease-wise model accuracy..." },
    { threshold: 70, label: "Rendering VGG19 pneumonia metrics..." },
    { threshold: 100, label: "Dashboard ready." },
  ];

  if (prefersReducedMotion) {
    updateLoader(fill, percent, status, 100, steps[3].label);
    loader.classList.add("is-complete");
    return;
  }

  let progress = 0;

  const timer = window.setInterval(() => {
    progress = Math.min(100, progress + 4 + Math.random() * 9);
    const currentStep =
      [...steps].reverse().find((step) => progress >= step.threshold) || steps[0];

    updateLoader(fill, percent, status, progress, currentStep.label);

    if (progress < 100) {
      return;
    }

    loader.classList.add("is-complete");
    window.clearInterval(timer);
  }, 85);
}

function updateLoader(fill, percent, status, progress, label) {
  fill.style.width = `${progress}%`;
  percent.textContent = `${Math.round(progress)}%`;
  status.textContent = label;
}

function setupProgressAnimations(prefersReducedMotion) {
  const fills = document.querySelectorAll("[data-progress]");

  if (!fills.length) {
    return;
  }

  const revealFill = (fill) => {
    if (fill.dataset.animated === "true") {
      return;
    }

    fill.dataset.animated = "true";
    const target = clampPercentage(fill.dataset.progress);
    fill.style.width = `${target}%`;

    const valueElement = fill.querySelector("[data-progress-value]");

    if (!valueElement) {
      return;
    }

    animateCount(valueElement, target, {
      suffix: valueElement.dataset.suffix || "%",
      precision: getPrecision(valueElement),
    });
  };

  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    fills.forEach(revealFill);
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        revealFill(entry.target);
        observer.unobserve(entry.target);
      });
    },
    {
      threshold: 0.24,
      rootMargin: "0px 0px -10% 0px",
    }
  );

  fills.forEach((fill) => {
    fill.style.width = "0%";
    observer.observe(fill);
  });
}

function setupCountups(prefersReducedMotion) {
  const counters = document.querySelectorAll("[data-countup]");

  if (!counters.length) {
    return;
  }

  const revealCounter = (counter) => {
    animateCount(counter, Number.parseFloat(counter.dataset.countup || "0"), {
      suffix: counter.dataset.suffix || "",
      precision: getPrecision(counter),
    });
  };

  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    counters.forEach(revealCounter);
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        revealCounter(entry.target);
        observer.unobserve(entry.target);
      });
    },
    {
      threshold: 0.3,
      rootMargin: "0px 0px -12% 0px",
    }
  );

  counters.forEach((counter) => observer.observe(counter));
}

function animateCount(element, target, options) {
  if (element.dataset.counted === "true") {
    return;
  }

  const safeTarget = Number.isFinite(target) ? target : 0;
  const precision = options.precision ?? 0;
  const suffix = options.suffix ?? "";

  element.dataset.counted = "true";

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    element.textContent = formatCount(safeTarget, precision, suffix);
    return;
  }

  const duration = 900;
  const start = performance.now();

  const step = (timestamp) => {
    const elapsed = Math.min((timestamp - start) / duration, 1);
    const eased = 1 - Math.pow(1 - elapsed, 3);
    element.textContent = formatCount(safeTarget * eased, precision, suffix);

    if (elapsed < 1) {
      window.requestAnimationFrame(step);
      return;
    }

    element.textContent = formatCount(safeTarget, precision, suffix);
  };

  window.requestAnimationFrame(step);
}

function formatCount(value, precision, suffix) {
  return `${value.toFixed(precision)}${suffix}`;
}

function getPrecision(element) {
  return Number.parseInt(element.dataset.precision || "0", 10);
}

function clampPercentage(value) {
  const parsed = Number.parseFloat(value || "0");

  if (!Number.isFinite(parsed)) {
    return 0;
  }

  return Math.max(0, Math.min(parsed, 100));
}
