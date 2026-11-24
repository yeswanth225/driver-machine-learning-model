import React, { useEffect, useId, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { cn } from "../../lib/utils";

const SIZES = {
  sm: { button: "h-8 gap-1.5 px-2.5 text-[12px]", square: "w-8", icon: 14 },
  default: { button: "h-8 sm:h-9 gap-2 px-3 text-xs sm:text-sm", square: "w-8 sm:w-9", icon: 15 },
  lg: { button: "h-10 gap-2 px-3.5 text-[14px]", square: "w-10", icon: 18 },
};

const EASE_OUT = "cubic-bezier(0.23, 1, 0.32, 1)";
const EASE_IN_OUT = "cubic-bezier(0.65, 0, 0.35, 1)";
const EASE_REVEAL = "cubic-bezier(0.4, 0, 0.2, 1)";
const REVEAL_ATTR = "data-theme-reveal";
const REVEAL_STYLE_ID = "kokonut-theme-reveal";
const REVEAL_CSS = `html[${REVEAL_ATTR}]::view-transition-old(root),html[${REVEAL_ATTR}]::view-transition-new(root){animation:none;mix-blend-mode:normal}`;

const RAY_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315];
const RAYS = RAY_ANGLES.map((deg) => {
  const rad = (deg * Math.PI) / 180;
  const at = (r) => ({
    x: Number((12 + r * Math.cos(rad)).toFixed(2)),
    y: Number((12 + r * Math.sin(rad)).toFixed(2)),
  });
  return { deg, from: at(7.5), to: at(10) };
});

const ANIMATED_PROPS = ["opacity", "scale", "translate", "filter"];

function snapshot(elements) {
  const map = new Map();
  for (const el of elements) {
    const style = getComputedStyle(el);
    const frame = { offset: 0 };
    for (const prop of ANIMATED_PROPS) {
      frame[prop] = style.getPropertyValue(prop);
    }
    map.set(el, frame);
    for (const animation of el.getAnimations()) {
      animation.cancel();
    }
  }
  return map;
}

function prefersReducedMotion() {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function ensureRevealStyle() {
  if (document.getElementById(REVEAL_STYLE_ID)) {
    return;
  }
  const style = document.createElement("style");
  style.id = REVEAL_STYLE_ID;
  style.textContent = REVEAL_CSS;
  document.head.appendChild(style);
}

function timing(el, toDark) {
  const { part, index = "0" } = el.dataset || {};
  const i = Number(index);
  switch (part) {
    case "ray":
      return {
        delay: toDark ? i * 18 : 140 + i * 22,
        duration: 260,
        easing: EASE_OUT,
      };
    case "bite":
      return { delay: toDark ? 60 : 0, duration: 380, easing: EASE_IN_OUT };
    case "disc":
      return { delay: toDark ? 40 : 0, duration: 380, easing: EASE_OUT };
    default:
      return { duration: 240, easing: EASE_OUT };
  }
}

function morph(from, toDark) {
  if (prefersReducedMotion()) {
    return;
  }
  for (const [el, frame] of from) {
    el.animate([frame, {}], { ...timing(el, toDark), fill: "backwards" });
  }
}

let revealRunning = false;

function revealFrom(button, apply) {
  const doc = document;
  if (
    typeof doc.startViewTransition !== "function" ||
    prefersReducedMotion() ||
    revealRunning ||
    doc.activeViewTransition
  ) {
    apply();
    return;
  }

  const rect = button.getBoundingClientRect();
  const x = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  const radius = Math.hypot(
    Math.max(x, window.innerWidth - x),
    Math.max(y, window.innerHeight - y)
  );
  const root = document.documentElement;

  ensureRevealStyle();
  root.setAttribute(REVEAL_ATTR, "");
  revealRunning = true;

  let transition;
  try {
    transition = doc.startViewTransition(() => flushSync(apply));
  } catch {
    root.removeAttribute(REVEAL_ATTR);
    revealRunning = false;
    apply();
    return;
  }

  transition.ready
    .then(() => {
      root.animate(
        {
          clipPath: [
            `circle(0px at ${x}px ${y}px)`,
            `circle(${radius}px at ${x}px ${y}px)`,
          ],
        },
        {
          duration: 500,
          easing: EASE_REVEAL,
          pseudoElement: "::view-transition-new(root)",
        }
      );
    })
    .catch(() => {});

  transition.finished
    .catch(() => {})
    .finally(() => {
      root.removeAttribute(REVEAL_ATTR);
      revealRunning = false;
    });
}

export function SwitchButton({
  className,
  size = "default",
  showLabel = true,
  theme,
  onThemeChange,
  reveal = true,
  onClick,
  ...props
}) {
  const maskId = `${useId()}-moon`;
  const iconRef = useRef(null);
  const labelRef = useRef(null);
  const [observed, setObserved] = useState(null);

  useEffect(() => {
    const root = document.documentElement;
    // Check saved local preference or existing class
    const saved = localStorage.getItem("theme");
    if (saved === "dark" && !root.classList.contains("dark")) {
      root.classList.add("dark");
      root.style.colorScheme = "dark";
    } else if (!saved && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches && !root.classList.contains("dark")) {
      root.classList.add("dark");
      root.style.colorScheme = "dark";
    }

    const read = () =>
      setObserved(root.classList.contains("dark") ? "dark" : "light");
    read();
    const observer = new MutationObserver(read);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  const current = observed && (theme ?? observed);
  const sizing = SIZES[size] || SIZES.default;

  const handleClick = (e) => {
    onClick?.(e);
    if (e.defaultPrevented) {
      return;
    }
    const root = document.documentElement;
    const from = theme ?? (root.classList.contains("dark") ? "dark" : "light");
    const next = from === "dark" ? "light" : "dark";
    const button = e.currentTarget;

    const apply = () => {
      const parts = [
        ...(iconRef.current?.querySelectorAll("[data-part]") ?? []),
        ...(labelRef.current?.querySelectorAll("[data-part]") ?? []),
      ];
      const before = snapshot(parts);
      root.classList.toggle("dark", next === "dark");
      root.style.colorScheme = next;
      try {
        localStorage.setItem("theme", next);
      } catch {}
      onThemeChange?.(next);
      morph(before, next === "dark");
    };

    if (reveal) {
      revealFrom(button, apply);
    } else {
      apply();
    }
  };

  return (
    <button
      aria-label="Toggle dark theme"
      aria-pressed={current ? current === "dark" : undefined}
      className={cn(
        "group/switch relative inline-flex shrink-0 cursor-pointer select-none items-center justify-center font-medium rounded-lg",
        "border border-[#dee2e6] bg-white text-black",
        "dark:border-[#27272a] dark:bg-[#0c0c0e] dark:text-white",
        "hover:border-black hover:bg-[#f8f9fa] dark:hover:border-white dark:hover:bg-[#18181b]",
        "active:scale-95",
        "transition-all duration-150 ease-out",
        "outline-none focus-visible:ring-2 focus-visible:ring-black dark:focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-white dark:focus-visible:ring-offset-black",
        "disabled:pointer-events-none disabled:opacity-50",
        sizing.button,
        !showLabel && cn(sizing.square, "px-0"),
        className
      )}
      type="button"
      {...props}
      onClick={handleClick}
    >
      <svg
        aria-hidden="true"
        className="pointer-events-none shrink-0 overflow-visible"
        fill="none"
        height={sizing.icon}
        ref={iconRef}
        viewBox="0 0 24 24"
        width={sizing.icon}
      >
        <mask id={maskId}>
          <rect fill="white" height="32" width="32" x="-4" y="-4" />
          <circle
            className="translate-x-[10px] -translate-y-[10px] dark:translate-x-0 dark:translate-y-0 transition-transform"
            cx="17"
            cy="7"
            data-part="bite"
            fill="black"
            r="7.5"
          />
        </mask>
        <g mask={`url(#${maskId})`}>
          <circle
            className="origin-[12px_12px] [transform-box:view-box] dark:scale-[1.8] transition-transform"
            cx="12"
            cy="12"
            data-part="disc"
            fill="currentColor"
            r="4.75"
          />
        </g>
        <g stroke="currentColor" strokeLinecap="round" strokeWidth="2">
          {RAYS.map((ray, index) => (
            <line
              className="origin-[12px_12px] [transform-box:view-box] dark:scale-50 dark:opacity-0 transition-all"
              data-index={index}
              data-part="ray"
              key={ray.deg}
              x1={ray.from.x}
              x2={ray.to.x}
              y1={ray.from.y}
              y2={ray.to.y}
            />
          ))}
        </g>
      </svg>
      {showLabel && (
        <span aria-hidden="true" className="grid leading-none font-display font-medium" ref={labelRef}>
          <span
            className="col-start-1 row-start-1 dark:opacity-0 dark:blur-[3px] transition-all"
            data-part="label"
          >
            Light
          </span>
          <span
            className="col-start-1 row-start-1 opacity-0 blur-[3px] dark:opacity-100 dark:blur-none transition-all"
            data-part="label"
          >
            Dark
          </span>
        </span>
      )}
    </button>
  );
}

export default SwitchButton;
