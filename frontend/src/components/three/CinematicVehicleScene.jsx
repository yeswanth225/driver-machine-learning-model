import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  Camera,
  Eye,
  Video,
  Compass,
  Shield,
  Palette,
  SunMedium,
  Info,
  Maximize2,
  Activity,
  Layers,
  Disc,
} from 'lucide-react';
import { Button } from '../ui/button';

/**
 * Premium Cinematic 3D Vehicle Simulation Scene.
 * Designed to evoke high-end automotive digital twin environments (inspired by 3dsimulation-eta.vercel.app).
 * Features:
 * - Raycasting click detection on vehicle components (chassis, wheels, lights, sensors, cabin)
 * - Hotspot inspection triggers
 * - Realistic studio 3-point lighting + rim light + contact shadow
 * - Event-reactive kinematics (pitch, roll, yaw, chassis vibration, brake flares, turn signals)
 * - Smooth camera transitions (Orbit, Chase, Cockpit/Hood, Tactical, Hero 3/4)
 * - Plain-language simulation disclaimer
 */
export function CinematicVehicleScene({
  safetyState = 'NORMAL',
  drivingStyle = 'Normal',
  maneuver = 'None / Nominal',
  riskScore = 0,
  confidence = 0.95,
  isLive = false,
  vehicleColor = 'silver', // silver | cyan | crimson | white | dark
  onColorChange,
  onInspectComponent,
  showControls = true,
  className = '',
}) {
  const containerRef = useRef(null);
  const [cameraMode, setCameraMode] = useState('orbit'); // orbit | chase | hood | tactical | hero
  const [quality, setQuality] = useState('high');
  const [activeColor, setActiveColor] = useState(vehicleColor);
  const [webGLError, setWebGLError] = useState(null);
  const [hoveredPart, setHoveredPart] = useState(null);

  const stateRef = useRef({
    safetyState,
    drivingStyle,
    maneuver,
    riskScore,
    cameraMode,
    quality,
    activeColor,
  });

  useEffect(() => {
    stateRef.current = {
      safetyState,
      drivingStyle,
      maneuver,
      riskScore,
      cameraMode,
      quality,
      activeColor,
    };
  }, [safetyState, drivingStyle, maneuver, riskScore, cameraMode, quality, activeColor]);

  // Color palette definitions for vehicle metallic finish
  const colorMap = {
    silver: { hex: 0xa8b4c4, name: 'Sterling Silver', desc: 'Liquid metallic silver with high reflectivity' },
    cyan: { hex: 0x0284c7, name: 'Cyber Azure', desc: 'Deep electric blue with cyan metallic flake' },
    crimson: { hex: 0xb91c1c, name: 'Crimson Red', desc: 'Sport red multi-coat finish' },
    white: { hex: 0xe2e8f0, name: 'Pearl White', desc: 'Triple-coat iridescent pearl white' },
    dark: { hex: 0x272d3b, name: 'Titanium Slate', desc: 'Dark graphite satin finish' },
  };

  const handleColorSelect = (cKey) => {
    setActiveColor(cKey);
    if (onColorChange) onColorChange(cKey);
  };

  const triggerComponentInspection = (partKey) => {
    if (!onInspectComponent) return;

    const currentStyle = stateRef.current.drivingStyle;
    const currentManeuver = stateRef.current.maneuver;
    const isAggressive = String(currentStyle).toLowerCase().includes('aggressive');

    switch (partKey) {
      case 'chassis':
        onInspectComponent({
          type: 'vehicle_component',
          data: {
            title: 'Vehicle Chassis & Aerodynamic Body',
            subtitle: colorMap[activeColor]?.name || 'Metallic Finish',
            description:
              'Lightweight rigid vehicle body constructed with high-strength alloys. Evaluated continuously for structural stability, lateral force resistance, and vibration dampening.',
            metrics: [
              { label: 'Exterior Finish', value: colorMap[activeColor]?.name || 'Silver', hint: 'Multi-coat metallic clearcoat' },
              { label: 'Chassis Motion', value: isAggressive ? 'Dynamic Shudder' : 'Quiescent Stable', hint: 'Body pitch & roll stability' },
              { label: 'Aerodynamic Drag', value: 'Cd 0.24', hint: 'Low frontal drag profile' },
              { label: 'Active Weight Balance', value: '50:50 Front / Rear', hint: 'Optimal traction distribution' },
            ],
            safetyNote: 'Chassis kinematics are cross-referenced with 6-DOF IMU telemetry to detect sudden lane swerves and roll over hazards.',
          },
        });
        break;

      case 'wheels':
        onInspectComponent({
          type: 'vehicle_component',
          data: {
            title: 'Wheels, Alloy Rims & Braking Calipers',
            subtitle: 'Front & Rear Axle Running Gear',
            description:
              'High-performance 5-spoke alloy wheels with hydraulic disc brake calipers. Rotational speed matches simulated vehicle velocity and reacts instantly to sudden braking deceleration.',
            metrics: [
              { label: 'Wheel Speed', value: currentManeuver.includes('BRAK') ? '10 m/s (Braking)' : '26 m/s (Cruising)', hint: 'Simulated rolling velocity' },
              { label: 'Brake Calipers', value: currentManeuver.includes('BRAK') ? 'High Clamp Pressure' : 'Ready Standby', hint: 'Hydraulic disc clamp' },
              { label: 'Steering Angle', value: currentManeuver.includes('LEFT') ? 'Left 32°' : currentManeuver.includes('RIGHT') ? 'Right 32°' : 'Center 0°', hint: 'Front wheel steering rack' },
              { label: 'Tire Grip State', value: isAggressive ? 'High Lateral Load' : 'Nominal Grip', hint: 'Pavement adhesion' },
            ],
            safetyNote: 'Rapid wheel deceleration beyond -0.4g immediately triggers the sudden deceleration safety protocol and taillight brake flare.',
          },
        });
        break;

      case 'lights':
        onInspectComponent({
          type: 'vehicle_component',
          data: {
            title: 'Intelligent LED Headlights & Taillights',
            subtitle: 'Forward Illumination & Dynamic Brake Flare',
            description:
              'Full LED lighting system with projector headlights casting a dynamic road beam and a full-width rear taillight bar that flares brightly during emergency braking.',
            metrics: [
              { label: 'Headlight Beam', value: 'High-Lumen Projector', hint: 'Illuminates virtual road corridor' },
              { label: 'Brake Taillight Flare', value: currentManeuver.includes('BRAK') ? '5.0× Emergency Flash' : 'Nominal Illumination', hint: 'Dynamic red emissive intensity' },
              { label: 'Turn Indicators', value: currentManeuver.includes('LEFT') ? 'Left Blinker (1.5 Hz)' : currentManeuver.includes('RIGHT') ? 'Right Blinker (1.5 Hz)' : 'Off', hint: 'Amber cornering signal' },
              { label: 'Underbody Glow', value: safetyState, hint: 'Safety status ground projection' },
            ],
            safetyNote: 'Dynamic brake light flaring alerts following traffic during sudden emergency braking maneuvers.',
          },
        });
        break;

      case 'sensors':
        onInspectComponent({
          type: 'sensor',
          data: {
            name: 'Vehicle 6-DOF Inertial Measurement Unit (IMU)',
            channel: 'Longitudinal, Lateral & Yaw Dynamics',
            explanation:
              'Centrally mounted sensor node combining a 3-axis linear accelerometer and a 3-axis angular rate gyroscope. Measures physical forces along vehicle length, width, and vertical rotational axes.',
            currentValue: currentManeuver.includes('BRAK') ? 'Acc X: -0.65 g (Braking)' : 'Acc X: 0.02 g (Cruising)',
            nominalRange: '± 0.25 g Normal Driving',
            rate: '10 Hz (Every 100 ms)',
            thresholdInfo:
              'Longitudinal forces exceeding -0.4g trigger Sudden Braking events. Lateral forces exceeding ±0.35g trigger Sharp Turn events. Windowed variances evaluate driving aggressiveness.',
          },
        });
        break;

      case 'cabin':
        onInspectComponent({
          type: 'vehicle_component',
          data: {
            title: 'Cabin & Driver Cockpit',
            subtitle: 'Driver Perspective & Passenger Safety Zone',
            description:
              'Acoustically insulated driver cabin featuring heads-up safety telemetry and optimized forward road visibility. The safety agent monitors overall vehicle motion to ensure passenger stability.',
            metrics: [
              { label: 'Driver Focus', value: 'Forward Road Corridor', hint: 'Cockpit line of sight' },
              { label: 'Passenger Comfort', value: isAggressive ? 'Elevated Lateral Forces' : 'Smooth & Stable', hint: 'Motion stability rating' },
              { label: 'Safety Guardrails', value: 'Active Monitoring', hint: 'Real-time hazard evaluation' },
              { label: 'Overall Safety', value: safetyState, hint: 'Current safety state' },
            ],
            safetyNote: 'In-cabin visual and acoustic advisories are dispatched when compound driving risk reaches Warning or High Risk thresholds.',
          },
        });
        break;

      default:
        break;
    }
  };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x090a0e); // Deep neutral obsidian
    scene.fog = new THREE.FogExp2(0x090a0e, 0.009);

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 500);
    camera.position.set(5.2, 2.8, 6.8);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: quality === 'high',
        powerPreference: 'high-performance',
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, quality === 'high' ? 2 : 1));
      renderer.shadowMap.enabled = quality === 'high';
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.35;
      container.appendChild(renderer.domElement);
    } catch (err) {
      console.error('WebGL initialization failed:', err);
      setWebGLError('3D Graphics Acceleration is currently unavailable.');
      return;
    }

    // 2. High-Visibility Automotive Studio Lighting
    const ambientLight = new THREE.AmbientLight(0x64748b, 1.8);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xe2e8f0, 0x1e293b, 1.6);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(8, 14, 10);
    if (quality === 'high') {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.width = 1024;
      keyLight.shadow.mapSize.height = 1024;
      keyLight.shadow.camera.near = 0.5;
      keyLight.shadow.camera.far = 40;
      keyLight.shadow.camera.left = -7;
      keyLight.shadow.camera.right = 7;
      keyLight.shadow.camera.top = 7;
      keyLight.shadow.camera.bottom = -7;
      keyLight.shadow.bias = -0.0004;
    }
    scene.add(keyLight);

    const frontFillLight = new THREE.DirectionalLight(0xf1f5f9, 2.0);
    frontFillLight.position.set(0, 5, 12);
    scene.add(frontFillLight);

    // Cyan electric rim light tracing vehicle silhouette
    const rimLight = new THREE.DirectionalLight(0x38bdf8, 2.8);
    rimLight.position.set(-10, 8, -12);
    scene.add(rimLight);

    const sideLight = new THREE.DirectionalLight(0x94a3b8, 1.8);
    sideLight.position.set(12, 4, -4);
    scene.add(sideLight);

    // 3. Environment & Virtual Road
    const roadGroup = new THREE.Group();
    scene.add(roadGroup);

    const roadWidth = 8.5;
    const roadLength = 130;
    const roadGeo = new THREE.PlaneGeometry(roadWidth, roadLength, 1, 1);
    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x141822, // Matte dark slate asphalt
      roughness: 0.8,
      metalness: 0.15,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.rotation.x = -Math.PI / 2;
    roadMesh.position.y = -0.01;
    roadMesh.receiveShadow = true;
    roadGroup.add(roadMesh);

    // Outer solid lane boundary stripes
    const stripeGeo = new THREE.PlaneGeometry(0.16, roadLength);
    const stripeMat = new THREE.MeshBasicMaterial({ color: 0x64748b });
    [-roadWidth / 2 + 0.35, roadWidth / 2 - 0.35].forEach((x) => {
      const stripe = new THREE.Mesh(stripeGeo, stripeMat);
      stripe.rotation.x = -Math.PI / 2;
      stripe.position.set(x, 0.005, 0);
      roadGroup.add(stripe);
    });

    // Road shoulder curbs
    const curbGeo = new THREE.BoxGeometry(0.35, 0.18, roadLength);
    const curbMat = new THREE.MeshStandardMaterial({ color: 0x1f2430, roughness: 0.7 });
    const leftCurb = new THREE.Mesh(curbGeo, curbMat);
    leftCurb.position.set(-roadWidth / 2 - 0.18, 0.06, 0);
    roadGroup.add(leftCurb);

    const rightCurb = new THREE.Mesh(curbGeo, curbMat);
    rightCurb.position.set(roadWidth / 2 + 0.18, 0.06, 0);
    roadGroup.add(rightCurb);

    // Scrolling Centerline Dashes (Bright white with glow)
    const dashCount = 28;
    const dashSpacing = 4.4;
    const dashGeo = new THREE.PlaneGeometry(0.2, 2.6);
    const dashMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    const dashes = [];
    for (let i = 0; i < dashCount; i++) {
      const dash = new THREE.Mesh(dashGeo, dashMat);
      dash.rotation.x = -Math.PI / 2;
      dash.position.set(0, 0.01, -roadLength / 2 + i * dashSpacing);
      roadGroup.add(dash);
      dashes.push(dash);
    }

    // Roadside Reflector Posts
    const reflectorPosts = [];
    const postGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.85, 8);
    const postMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.3 });
    const postReflectMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });

    for (let i = 0; i < 18; i++) {
      const zPos = -roadLength / 2 + i * (roadLength / 18);
      [-roadWidth / 2 - 0.85, roadWidth / 2 + 0.85].forEach((xPos) => {
        const post = new THREE.Mesh(postGeo, postMat);
        post.position.set(xPos, 0.42, zPos);
        const reflector = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 0.05), postReflectMat);
        reflector.position.set(0, 0.32, 0);
        post.add(reflector);
        roadGroup.add(post);
        reflectorPosts.push(post);
      });
    }

    // 4. Procedural High-Visibility Sports Vehicle Digital Twin
    const carRoot = new THREE.Group();
    scene.add(carRoot);

    const initialHex = colorMap[activeColor]?.hex || 0xa8b4c4;
    const bodyPaintMat = new THREE.MeshStandardMaterial({
      color: initialHex,
      metalness: 0.85,
      roughness: 0.18,
    });

    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x0c4a6e,
      metalness: 0.9,
      roughness: 0.08,
      transparent: true,
      opacity: 0.82,
    });

    const trimCarbonMat = new THREE.MeshStandardMaterial({
      color: 0x161b24,
      roughness: 0.45,
      metalness: 0.6,
    });

    const chromeAlloyMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      metalness: 0.95,
      roughness: 0.08,
    });

    const brakeCaliperMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      roughness: 0.3,
    });

    const headlightMat = new THREE.MeshBasicMaterial({ color: 0xf8fafc });
    const taillightMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xef4444,
      emissiveIntensity: 1.2,
      roughness: 0.2,
    });

    const indicatorLeftMat = new THREE.MeshBasicMaterial({ color: 0x334155 });
    const indicatorRightMat = new THREE.MeshBasicMaterial({ color: 0x334155 });

    const underglowMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.45,
    });

    // Sensor Beacon Material (Pulsing Cyan)
    const sensorBeaconMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.9,
    });

    // Array of interactive meshes for raycasting
    const interactiveMeshes = [];

    // Lower Chassis
    const chassisGeo = new THREE.BoxGeometry(1.94, 0.44, 4.35);
    const chassis = new THREE.Mesh(chassisGeo, bodyPaintMat);
    chassis.position.y = 0.46;
    chassis.castShadow = true;
    chassis.receiveShadow = true;
    chassis.userData = { partKey: 'chassis' };
    carRoot.add(chassis);
    interactiveMeshes.push(chassis);

    // Front aerodynamic hood
    const hoodGeo = new THREE.BoxGeometry(1.86, 0.22, 1.45);
    const hood = new THREE.Mesh(hoodGeo, bodyPaintMat);
    hood.position.set(0, 0.64, 1.25);
    hood.rotation.x = -0.06;
    hood.castShadow = true;
    hood.userData = { partKey: 'chassis' };
    carRoot.add(hood);
    interactiveMeshes.push(hood);

    // Front grille trim
    const grilleGeo = new THREE.BoxGeometry(1.4, 0.16, 0.1);
    const grille = new THREE.Mesh(grilleGeo, trimCarbonMat);
    grille.position.set(0, 0.42, 2.18);
    carRoot.add(grille);

    // Cabin Greenhouse & Glass
    const cabinGeo = new THREE.BoxGeometry(1.62, 0.50, 2.15);
    const cabin = new THREE.Mesh(cabinGeo, glassMat);
    cabin.position.set(0, 0.90, -0.18);
    cabin.castShadow = true;
    cabin.userData = { partKey: 'cabin' };
    carRoot.add(cabin);
    interactiveMeshes.push(cabin);

    // Cabin Roof Panel
    const roofGeo = new THREE.BoxGeometry(1.56, 0.06, 1.55);
    const roof = new THREE.Mesh(roofGeo, bodyPaintMat);
    roof.position.set(0, 1.16, -0.22);
    roof.castShadow = true;
    roof.userData = { partKey: 'cabin' };
    carRoot.add(roof);
    interactiveMeshes.push(roof);

    // 6-DOF IMU Sensor Beacon on Roof Center
    const beaconGeo = new THREE.CylinderGeometry(0.12, 0.16, 0.1, 16);
    const beacon = new THREE.Mesh(beaconGeo, sensorBeaconMat);
    beacon.position.set(0, 1.22, -0.22);
    beacon.userData = { partKey: 'sensors' };
    carRoot.add(beacon);
    interactiveMeshes.push(beacon);

    // Sensor Beacon Pulse Ring
    const beaconRingGeo = new THREE.RingGeometry(0.18, 0.26, 16);
    const beaconRingMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.6,
    });
    const beaconRing = new THREE.Mesh(beaconRingGeo, beaconRingMat);
    beaconRing.rotation.x = -Math.PI / 2;
    beaconRing.position.set(0, 1.27, -0.22);
    carRoot.add(beaconRing);

    // Rear Diffuser
    const diffuserGeo = new THREE.BoxGeometry(1.88, 0.24, 0.55);
    const diffuser = new THREE.Mesh(diffuserGeo, trimCarbonMat);
    diffuser.position.set(0, 0.36, -2.12);
    carRoot.add(diffuser);

    // Dual Projector LED Headlights + Ground Projection
    [-0.70, 0.70].forEach((x) => {
      const hlMesh = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.09, 0.16), headlightMat);
      hlMesh.position.set(x, 0.56, 2.18);
      hlMesh.userData = { partKey: 'lights' };
      carRoot.add(hlMesh);
      interactiveMeshes.push(hlMesh);

      // Forward light beam cone
      const beamGeo = new THREE.ConeGeometry(0.9, 5.0, 16, 1, true);
      const beamMat = new THREE.MeshBasicMaterial({
        color: 0x93c5fd,
        transparent: true,
        opacity: 0.12,
        side: THREE.DoubleSide,
      });
      const beam = new THREE.Mesh(beamGeo, beamMat);
      beam.rotation.x = -Math.PI / 2;
      beam.position.set(x, 0.56, 4.6);
      carRoot.add(beam);

      // Dynamic forward road spotlight
      const spotLight = new THREE.SpotLight(0xffffff, 2.5, 18, Math.PI / 6, 0.4, 1);
      spotLight.position.set(x, 0.6, 2.2);
      spotLight.target.position.set(x * 0.5, 0, 12);
      scene.add(spotLight);
      scene.add(spotLight.target);
    });

    // Rear Full-Width LED Taillight Strip
    const taillightBarGeo = new THREE.BoxGeometry(1.76, 0.10, 0.12);
    const taillightBar = new THREE.Mesh(taillightBarGeo, taillightMat);
    taillightBar.position.set(0, 0.60, -2.18);
    taillightBar.userData = { partKey: 'lights' };
    carRoot.add(taillightBar);
    interactiveMeshes.push(taillightBar);

    // Turn Signal Indicators (Amber)
    const turnLeftGeo = new THREE.BoxGeometry(0.22, 0.07, 0.09);
    const turnLeftFront = new THREE.Mesh(turnLeftGeo, indicatorLeftMat);
    turnLeftFront.position.set(-0.88, 0.56, 2.16);
    carRoot.add(turnLeftFront);

    const turnLeftRear = new THREE.Mesh(turnLeftGeo, indicatorLeftMat);
    turnLeftRear.position.set(-0.88, 0.60, -2.16);
    carRoot.add(turnLeftRear);

    const turnRightGeo = new THREE.BoxGeometry(0.22, 0.07, 0.09);
    const turnRightFront = new THREE.Mesh(turnRightGeo, indicatorRightMat);
    turnRightFront.position.set(0.88, 0.56, 2.16);
    carRoot.add(turnRightFront);

    const turnRightRear = new THREE.Mesh(turnRightGeo, indicatorRightMat);
    turnRightRear.position.set(0.88, 0.60, -2.16);
    carRoot.add(turnRightRear);

    // Wheels & Steering Hubs with Alloy Rims & Calipers
    const wheelGeo = new THREE.CylinderGeometry(0.38, 0.38, 0.28, 28);
    wheelGeo.rotateZ(Math.PI / 2);
    const tireMat = new THREE.MeshStandardMaterial({ color: 0x1a1f28, roughness: 0.85 });

    const rimGeo = new THREE.CylinderGeometry(0.26, 0.26, 0.29, 14);
    rimGeo.rotateZ(Math.PI / 2);

    const caliperGeo = new THREE.BoxGeometry(0.12, 0.18, 0.12);

    const frontLeftHub = new THREE.Group();
    frontLeftHub.position.set(-0.98, 0.38, 1.25);
    carRoot.add(frontLeftHub);

    const frontRightHub = new THREE.Group();
    frontRightHub.position.set(0.98, 0.38, 1.25);
    carRoot.add(frontRightHub);

    const createWheel = (parent, isLeft) => {
      const tire = new THREE.Mesh(wheelGeo, tireMat);
      tire.castShadow = true;
      tire.userData = { partKey: 'wheels' };
      interactiveMeshes.push(tire);

      const rim = new THREE.Mesh(rimGeo, chromeAlloyMat);
      rim.userData = { partKey: 'wheels' };
      tire.add(rim);

      const caliper = new THREE.Mesh(caliperGeo, brakeCaliperMat);
      caliper.position.set(isLeft ? 0.05 : -0.05, 0.12, 0);
      caliper.userData = { partKey: 'wheels' };
      parent.add(caliper);

      parent.add(tire);
      return tire;
    };

    const wheelFL = createWheel(frontLeftHub, true);
    const wheelFR = createWheel(frontRightHub, false);

    const rearLeftHub = new THREE.Group();
    rearLeftHub.position.set(-0.98, 0.38, -1.25);
    carRoot.add(rearLeftHub);
    const wheelRL = createWheel(rearLeftHub, true);

    const rearRightHub = new THREE.Group();
    rearRightHub.position.set(0.98, 0.38, -1.25);
    carRoot.add(rearRightHub);
    const wheelRR = createWheel(rearRightHub, false);

    const wheels = [wheelFL, wheelFR, wheelRL, wheelRR];

    // Underbody Ground Glow
    const underglowGeo = new THREE.PlaneGeometry(2.4, 4.8);
    const underglowMesh = new THREE.Mesh(underglowGeo, underglowMat);
    underglowMesh.rotation.x = -Math.PI / 2;
    underglowMesh.position.y = 0.03;
    carRoot.add(underglowMesh);

    // 5. Interactive Raycaster for Clicking & Hovering on Components
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onPointerClick = (e) => {
      const rect = container.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(interactiveMeshes, true);

      if (intersects.length > 0) {
        let currentObj = intersects[0].object;
        while (currentObj && !currentObj.userData?.partKey && currentObj.parent) {
          currentObj = currentObj.parent;
        }
        if (currentObj?.userData?.partKey) {
          triggerComponentInspection(currentObj.userData.partKey);
        }
      }
    };

    container.addEventListener('click', onPointerClick);

    // 6. Interactive Orbit & Mouse Drag Controls
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let spherical = { radius: 8.5, theta: 0.65, phi: 1.22 };

    const onMouseDown = (e) => {
      if (stateRef.current.cameraMode !== 'orbit') return;
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseMove = (e) => {
      if (!isDragging || stateRef.current.cameraMode !== 'orbit') return;
      const deltaX = e.clientX - prevMouseX;
      const deltaY = e.clientY - prevMouseY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;

      spherical.theta -= deltaX * 0.008;
      spherical.phi = Math.max(0.2, Math.min(Math.PI / 2 - 0.05, spherical.phi - deltaY * 0.008));
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (e) => {
      if (stateRef.current.cameraMode !== 'orbit') return;
      e.preventDefault();
      spherical.radius = Math.max(4.0, Math.min(18.0, spherical.radius + e.deltaY * 0.01));
    };

    container.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    container.addEventListener('wheel', onWheel, { passive: false });

    // 7. Animation Physics & Kinematics Loop
    let animationFrameId;
    let clock = new THREE.Clock();
    let roadSpeed = 26.0;
    let currentCarX = 0;
    let currentCarPitch = 0;
    let currentCarRoll = 0;
    let currentSteerAngle = 0;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsedTime = clock.getElapsedTime();
      const current = stateRef.current;

      // Update vehicle body paint dynamically
      const desiredColor = colorMap[current.activeColor]?.hex || 0xa8b4c4;
      if (bodyPaintMat.color.getHex() !== desiredColor) {
        bodyPaintMat.color.setHex(desiredColor);
      }

      // Sensor Beacon Pulse Animation
      const pulseScale = 1.0 + Math.sin(elapsedTime * 4.0) * 0.35;
      beaconRing.scale.set(pulseScale, pulseScale, 1.0);
      beaconRingMat.opacity = Math.max(0.1, 0.6 - Math.sin(elapsedTime * 4.0) * 0.4);

      // Driving style and event detection
      const isAggressive = String(current.drivingStyle).toLowerCase().includes('aggressive');
      const maneuverStr = String(current.maneuver).toUpperCase();

      const isSA = maneuverStr.includes('ACCELERATION') || maneuverStr === 'SA';
      const isSD = maneuverStr.includes('DECELERATION') || maneuverStr.includes('BRAK') || maneuverStr === 'SD';
      const isSLT = maneuverStr.includes('LEFT') || maneuverStr === 'SLT';
      const isSRT = maneuverStr.includes('RIGHT') || maneuverStr === 'SRT';

      // Speed modulation
      let targetSpeed = 26.0;
      if (isSA) targetSpeed = 44.0;
      else if (isSD) targetSpeed = 10.0;
      else if (isAggressive) targetSpeed = 33.0;

      roadSpeed = THREE.MathUtils.lerp(roadSpeed, targetSpeed, delta * 3.5);

      // Scroll Road Dashes
      const moveDistance = roadSpeed * delta;
      dashes.forEach((dash) => {
        dash.position.z -= moveDistance;
        if (dash.position.z < -roadLength / 2) {
          dash.position.z += roadLength;
        }
      });

      reflectorPosts.forEach((post) => {
        post.position.z -= moveDistance;
        if (post.position.z < -roadLength / 2) {
          post.position.z += roadLength;
        }
      });

      // Spin Wheels
      const spinSpeed = (roadSpeed / 0.38) * delta;
      wheels.forEach((w) => w.rotateX(spinSpeed));

      // Dynamic Vehicle Suspension & Reaction Poses
      let targetPitch = 0;
      let targetRoll = 0;
      let targetSteer = 0;
      let targetCarX = 0;
      let suspensionBob = Math.sin(elapsedTime * (isAggressive ? 9 : 5)) * (isAggressive ? 0.024 : 0.012);

      if (isSA) {
        targetPitch = -0.065; // Nose up / rear squat
        suspensionBob += (Math.random() - 0.5) * 0.01;
      } else if (isSD) {
        targetPitch = 0.085; // Nose dive
      }

      if (isSLT) {
        targetSteer = 0.32; // Front wheels steer left
        targetRoll = 0.09; // Centrifugal roll right
        targetCarX = -1.2;
      } else if (isSRT) {
        targetSteer = -0.32; // Front wheels steer right
        targetRoll = -0.09; // Centrifugal roll left
        targetCarX = 1.2;
      } else if (isAggressive) {
        targetCarX = Math.sin(elapsedTime * 2.2) * 0.35;
      }

      currentCarPitch = THREE.MathUtils.lerp(currentCarPitch, targetPitch, delta * 6);
      currentCarRoll = THREE.MathUtils.lerp(currentCarRoll, targetRoll, delta * 6);
      currentSteerAngle = THREE.MathUtils.lerp(currentSteerAngle, targetSteer, delta * 8);
      currentCarX = THREE.MathUtils.lerp(currentCarX, targetCarX, delta * 4);

      carRoot.position.x = currentCarX;
      carRoot.position.y = suspensionBob;
      carRoot.rotation.x = currentCarPitch;
      carRoot.rotation.z = currentCarRoll;

      frontLeftHub.rotation.y = currentSteerAngle;
      frontRightHub.rotation.y = currentSteerAngle;

      // Taillights Flare on Sudden Braking
      if (isSD) {
        taillightMat.emissiveIntensity = 5.0;
        taillightMat.emissive.setHex(0xff0000);
      } else {
        taillightMat.emissiveIntensity = 1.2;
        taillightMat.emissive.setHex(0xef4444);
      }

      // Amber Turn Indicators Blinking (1.5 Hz)
      const blinkCycle = Math.sin(elapsedTime * 9.5) > 0;
      if (isSLT && blinkCycle) {
        indicatorLeftMat.color.setHex(0xf59e0b);
      } else {
        indicatorLeftMat.color.setHex(0x334155);
      }

      if (isSRT && blinkCycle) {
        indicatorRightMat.color.setHex(0xf59e0b);
      } else {
        indicatorRightMat.color.setHex(0x334155);
      }

      // Dynamic Underglow Status Color
      const state = String(current.safetyState).toUpperCase();
      if (state === 'CRITICAL' || state === 'HIGH_RISK') {
        underglowMat.color.setHex(0xef4444);
        underglowMat.opacity = 0.75;
      } else if (state === 'WARNING') {
        underglowMat.color.setHex(0xf43f5e);
        underglowMat.opacity = 0.6;
      } else if (state === 'CAUTION') {
        underglowMat.color.setHex(0xf59e0b);
        underglowMat.opacity = 0.5;
      } else {
        underglowMat.color.setHex(0x10b981);
        underglowMat.opacity = 0.4;
      }

      // Camera Positioning Modes
      if (current.cameraMode === 'chase') {
        const targetCamPos = new THREE.Vector3(currentCarX * 0.6, 2.3, -6.8);
        camera.position.lerp(targetCamPos, delta * 5);
        camera.lookAt(currentCarX * 0.4, 0.75, 4.0);
      } else if (current.cameraMode === 'hood') {
        camera.position.set(currentCarX, 1.05, 0.45);
        camera.lookAt(currentCarX + currentSteerAngle * 2.0, 0.95, 15.0);
      } else if (current.cameraMode === 'tactical') {
        const targetCamPos = new THREE.Vector3(7.2, 8.5, 4.2);
        camera.position.lerp(targetCamPos, delta * 4);
        camera.lookAt(currentCarX, 0.5, 0);
      } else if (current.cameraMode === 'hero') {
        // High-end 3/4 front beauty angle
        const heroPos = new THREE.Vector3(4.8, 1.9, 5.2);
        camera.position.lerp(heroPos, delta * 4);
        camera.lookAt(currentCarX * 0.5, 0.6, 0);
      } else {
        // Orbit Mode
        const cx = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
        const cy = spherical.radius * Math.cos(spherical.phi);
        const cz = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
        camera.position.set(cx + currentCarX * 0.5, cy, cz);
        camera.lookAt(currentCarX * 0.5, 0.6, 0);
      }

      renderer.render(scene, camera);
    };

    animate();

    // 8. Handle Resize
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };
    window.addEventListener('resize', handleResize);

    // 9. Cleanup & Disposal
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      container.removeEventListener('click', onPointerClick);
      container.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      container.removeEventListener('wheel', onWheel);

      scene.traverse((obj) => {
        if (obj.isMesh) {
          obj.geometry?.dispose();
          if (Array.isArray(obj.material)) {
            obj.material.forEach((m) => m?.dispose());
          } else {
            obj.material?.dispose();
          }
        }
      });

      if (renderer?.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer?.dispose();
    };
  }, [quality]);

  if (webGLError) {
    return (
      <div className={`relative flex items-center justify-center bg-[#0e1118] border border-neutral-800 rounded-xl p-8 text-center ${className}`}>
        <div className="max-w-md">
          <Shield className="w-10 h-10 text-amber-400 mx-auto mb-3" />
          <h3 className="text-base font-semibold text-neutral-100">3D Vehicle View Offline</h3>
          <p className="text-xs text-neutral-400 mt-1">{webGLError}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative w-full h-full overflow-hidden rounded-sm border border-[#222226] bg-black select-none ${className}`}>
      {/* Three.js Canvas Container */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Top Left Status & Title HUD */}
      <div className="absolute top-3.5 left-4 flex flex-col sm:flex-row items-start sm:items-center gap-2 pointer-events-none">
        <div className="flex items-center gap-2 px-3 py-1 bg-black/90 border border-[#222226] rounded-full">
          <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-white' : 'bg-[#5a5a5f]'}`} />
          <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-white">
            {isLive ? 'Live Telemetry' : '3D Vehicle Simulation'}
          </span>
        </div>

        <div className="px-3 py-1 bg-black/90 border border-[#222226] rounded-full font-display font-bold uppercase tracking-spacex text-[10px] text-[#8a8a90] hidden md:block">
          Studio Perspective // 60 FPS
        </div>
      </div>

      {/* Top Right Controls: Camera Modes & Vehicle Finish */}
      {showControls && (
        <div className="absolute top-3.5 right-4 flex flex-wrap items-center justify-end gap-2">
          {/* Vehicle Paint Switcher */}
          <div className="flex items-center gap-1.5 bg-black/90 border border-[#222226] rounded-full px-2.5 py-1">
            <span className="text-[10px] font-display font-bold uppercase tracking-spacex text-[#8a8a90] pr-1 hidden lg:inline">
              Hull //
            </span>
            {Object.entries(colorMap).map(([key, col]) => (
              <button
                key={key}
                onClick={() => handleColorSelect(key)}
                title={col.name}
                className={`w-3.5 h-3.5 rounded-full border transition-transform ${
                  activeColor === key ? 'scale-125 border-white ring-1 ring-white' : 'border-[#3a3a3f] opacity-60 hover:opacity-100'
                }`}
                style={{ backgroundColor: `#${col.hex.toString(16).padStart(6, '0')}` }}
              />
            ))}
          </div>

          {/* Camera Mode Selector */}
          <div className="flex items-center gap-1 bg-black/80 backdrop-blur-md border border-white/20 rounded-full p-0.5">
            <Button
              variant={cameraMode === 'orbit' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setCameraMode('orbit')}
              title="Free Orbit Camera"
              className={`px-3 py-1 rounded-full text-[11px] font-display transition-all h-7 gap-1.5 ${
                cameraMode === 'orbit' ? 'bg-[#0066cc] text-white hover:bg-[#0052a3]' : 'text-zinc-300 hover:text-white'
              }`}
            >
              <Camera className="w-3 h-3" />
              <span className="hidden sm:inline">Orbit</span>
            </Button>

            <Button
              variant={cameraMode === 'chase' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setCameraMode('chase')}
              title="Chase Camera"
              className={`px-3 py-1 rounded-full text-[11px] font-display transition-all h-7 gap-1.5 ${
                cameraMode === 'chase' ? 'bg-[#0066cc] text-white hover:bg-[#0052a3]' : 'text-zinc-300 hover:text-white'
              }`}
            >
              <Video className="w-3 h-3" />
              <span className="hidden sm:inline">Chase</span>
            </Button>

            <Button
              variant={cameraMode === 'hood' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setCameraMode('hood')}
              title="Cockpit Perspective"
              className={`px-3 py-1 rounded-full text-[11px] font-display transition-all h-7 gap-1.5 ${
                cameraMode === 'hood' ? 'bg-[#0066cc] text-white hover:bg-[#0052a3]' : 'text-zinc-300 hover:text-white'
              }`}
            >
              <Eye className="w-3 h-3" />
              <span className="hidden sm:inline">Cockpit</span>
            </Button>

            <Button
              variant={cameraMode === 'tactical' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setCameraMode('tactical')}
              title="Tactical Isometric View"
              className={`px-3 py-1 rounded-full text-[11px] font-display transition-all h-7 gap-1.5 ${
                cameraMode === 'tactical' ? 'bg-[#0066cc] text-white hover:bg-[#0052a3]' : 'text-zinc-300 hover:text-white'
              }`}
            >
              <Compass className="w-3 h-3" />
              <span className="hidden sm:inline">Tactical</span>
            </Button>

            <Button
              variant={cameraMode === 'hero' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setCameraMode('hero')}
              title="Hero Perspective"
              className={`px-3 py-1 rounded-full text-[11px] font-display transition-all h-7 gap-1.5 ${
                cameraMode === 'hero' ? 'bg-[#0066cc] text-white hover:bg-[#0052a3]' : 'text-zinc-300 hover:text-white'
              }`}
            >
              <Eye className="w-3 h-3" />
              <span className="hidden sm:inline">Hero</span>
            </Button>
          </div>
        </div>
      )}

      {/* Interactive Hotspot Inspector Pills */}
      <div className="absolute top-14 left-4 flex flex-wrap items-center gap-1.5 z-20">
        <span className="text-[11px] font-display text-zinc-400 bg-black/80 backdrop-blur-md px-2.5 py-0.5 rounded-full border border-white/20">
          Inspect:
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => triggerComponentInspection('chassis')}
          className="px-2.5 py-0.5 h-6 rounded-full bg-black/80 backdrop-blur-md hover:bg-[#0066cc] hover:border-[#0066cc] border border-white/20 text-[11px] font-display text-white transition-all active:scale-95"
        >
          Chassis
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => triggerComponentInspection('wheels')}
          className="px-2.5 py-0.5 h-6 rounded-full bg-black/80 backdrop-blur-md hover:bg-[#0066cc] hover:border-[#0066cc] border border-white/20 text-[11px] font-display text-white transition-all active:scale-95"
        >
          Brakes &amp; Wheels
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => triggerComponentInspection('lights')}
          className="px-2.5 py-0.5 h-6 rounded-full bg-black/80 backdrop-blur-md hover:bg-[#0066cc] hover:border-[#0066cc] border border-white/20 text-[11px] font-display text-white transition-all active:scale-95"
        >
          Lighting
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => triggerComponentInspection('sensors')}
          className="px-2.5 py-0.5 h-6 rounded-full bg-black/80 backdrop-blur-md hover:bg-[#0066cc] hover:border-[#0066cc] border border-[#2997ff] text-[11px] font-display text-white transition-all active:scale-95"
        >
          6-DOF IMU
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => triggerComponentInspection('cabin')}
          className="px-2.5 py-0.5 h-6 rounded-full bg-black/80 backdrop-blur-md hover:bg-[#0066cc] hover:border-[#0066cc] border border-white/20 text-[11px] font-display text-white transition-all active:scale-95"
        >
          Cabin
        </Button>
      </div>

      {/* Floating Bottom Telemetry & Status Bar */}
      <div className="absolute bottom-9 inset-x-4 flex items-center justify-between pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="px-3.5 py-1.5 bg-black/80 backdrop-blur-md border border-white/15 rounded-[12px]">
            <span className="text-[10px] font-display text-zinc-400 block">Behavior:</span>
            <span className={`text-xs font-display font-medium ${drivingStyle.toLowerCase().includes('aggressive') ? 'text-amber-400' : 'text-white'}`}>
              {drivingStyle.toLowerCase().includes('aggressive') ? 'Aggressive Dynamics' : 'Nominal Cruising'}
            </span>
          </div>

          <div className="px-3.5 py-1.5 bg-black/80 backdrop-blur-md border border-white/15 rounded-[12px]">
            <span className="text-[10px] font-display text-zinc-400 block">Driving Maneuver:</span>
            <span className="text-xs font-display font-medium text-white">
              {maneuver}
            </span>
          </div>
        </div>

        <div className="px-3.5 py-1.5 bg-black/80 backdrop-blur-md border border-white/15 rounded-[12px] text-right pointer-events-auto">
          <span className="text-[10px] font-display text-zinc-400 block">Safety State:</span>
          <span className="text-xs font-display font-medium text-white">
            {safetyState}
          </span>
        </div>
      </div>

      {/* Simulation Disclaimer Bar at Bottom */}
      <div className="absolute bottom-0 inset-x-0 bg-black/90 border-t border-white/10 py-1.5 px-4 flex items-center justify-between text-[11px] font-display text-zinc-400 pointer-events-none">
        <div className="truncate">
          <span className="text-white font-medium mr-1.5">Digital Twin Telemetry:</span>
          <span>
            Real-time 3D simulation driven by deterministic Safety Agent inference.
          </span>
        </div>
        <span className="text-zinc-500 shrink-0 hidden sm:inline">
          Click any vehicle component to inspect sensor telemetry
        </span>
      </div>
    </div>
  );
}
