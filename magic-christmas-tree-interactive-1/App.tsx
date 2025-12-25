
import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment';
import { FilesetResolver, HandLandmarker, HandLandmarkerResult } from '@mediapipe/tasks-vision';

// --- Constants ---
const CONFIG = {
  goldCount: 500,
  silverCount: 500,
  gemCount: 300,
  emeraldCount: 300,
  dustCount: 1000,
  treeHeight: 75,
  maxRadius: 30,
  camDistance: 125,
  bloomStrength: 0.7,
  bloomThreshold: 0.35,
  bloomRadius: 0.6
};

enum STATE { TREE = 'tree', SCATTER = 'scatter', ZOOM = 'zoom', ORBIT = 'orbit' }

// Adjusted for LOWER brightness per user request
// 0x333333 is quite dark for inactive photos
const COLOR_DIM = new THREE.Color(0x333333);
// 0x999999 ensures even "active" photos aren't fully white/bright
const COLOR_FULL = new THREE.Color(0x999999); 

// Helper types
interface ParticleData {
  treePos: THREE.Vector3;
  scatterPos: THREE.Vector3;
  currentPos: THREE.Vector3;
  scale: number;
  rotSpeed: THREE.Euler;
  rotation: THREE.Euler;
}

interface DustData {
  treePos: THREE.Vector3;
  scatterPos: THREE.Vector3;
  currentPos: THREE.Vector3;
  velocity: number;
}

const App: React.FC = () => {
  // Refs for DOM elements
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const skeletonCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Scene variables (stored in refs to avoid re-renders interrupting Three.js loop)
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const composerRef = useRef<EffectComposer | null>(null);
  const mainGroupRef = useRef<THREE.Group>(new THREE.Group());
  const photoMeshesRef = useRef<THREE.Mesh[]>([]);
  const logicDataRef = useRef<{
    gold: ParticleData[];
    silver: ParticleData[];
    gem: ParticleData[];
    emerald: ParticleData[];
    dust: DustData[];
    star: THREE.Mesh | null;
  }>({ gold: [], silver: [], gem: [], emerald: [], dust: [], star: null });

  const meshesRef = useRef<{
    gold: THREE.InstancedMesh | null;
    silver: THREE.InstancedMesh | null;
    gem: THREE.InstancedMesh | null;
    emerald: THREE.InstancedMesh | null;
    dust: THREE.Points | null;
    bgParticles: THREE.Points | null;
  }>({ gold: null, silver: null, gem: null, emerald: null, dust: null, bgParticles: null });

  // State control
  const currentStateRef = useRef<STATE>(STATE.TREE);
  const [status, setStatus] = useState<string>("Initializing Engine...");
  const [statusColor, setStatusColor] = useState<string>("#fff");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [zoomTargetIndex, setZoomTargetIndex] = useState<number>(0);
  const zoomTargetIndexRef = useRef<number>(0);

  // Interaction refs
  const handPosRef = useRef({ x: 0, y: 0 });
  const lastHandPosRef = useRef({ x: 0, y: 0 });
  const isHandPresentRef = useRef(false);
  const rotationVelocityRef = useRef({ x: 0, y: 0 });
  const wasPinchRef = useRef(false);
  const timeRef = useRef(0);

  // --- Hand Tracking Utils ---
  const handleGesture = useCallback((landmarks: any[]) => {
    const palmX = 1 - (landmarks[0].x + landmarks[9].x) / 2;
    const palmY = (landmarks[0].y + landmarks[9].y) / 2;

    handPosRef.current.x = handPosRef.current.x * 0.7 + palmX * 0.3;
    handPosRef.current.y = handPosRef.current.y * 0.7 + palmY * 0.3;

    const tips = [4, 8, 12, 16, 20];
    const wrist = landmarks[0];
    const folded = tips.map((tipIdx, i) => {
      const d = Math.sqrt(Math.pow(landmarks[tipIdx].x - wrist.x, 2) + Math.pow(landmarks[tipIdx].y - wrist.y, 2));
      const threshold = i === 0 ? 0.3 : 0.25;
      return d < threshold;
    });

    const dPinch = Math.sqrt(Math.pow(landmarks[4].x - landmarks[8].x, 2) + Math.pow(landmarks[4].y - landmarks[8].y, 2));
    const isPinch = dPinch < 0.08;
    const isFist = folded[1] && folded[2] && folded[3] && folded[4];
    const isPeace = !folded[1] && !folded[2] && folded[3] && folded[4] && !isPinch;
    const isOpen = !isFist && !isPinch && !isPeace;

    if (isPinch) {
      if (!wasPinchRef.current) {
        if (photoMeshesRef.current.length > 0) {
          currentStateRef.current = STATE.ZOOM;
          const nextIdx = (zoomTargetIndexRef.current + 1) % photoMeshesRef.current.length;
          zoomTargetIndexRef.current = nextIdx;
          setZoomTargetIndex(nextIdx);
          setStatus(`Showing Photo ${nextIdx + 1} / ${photoMeshesRef.current.length}`);
          setStatusColor("#ffd700");
        } else {
          setStatus("Please upload photos first");
        }
      }
    } else if (isPeace) {
      if (photoMeshesRef.current.length > 0) {
        currentStateRef.current = STATE.ORBIT;
        setStatus("Wish Tree Mode (Auto Rotating)");
        setStatusColor("#ff00ff");
      } else {
        setStatus("Peace gesture: No photos");
      }
    } else if (isFist) {
      currentStateRef.current = STATE.TREE;
      setStatus("Fist: Tree Form");
      setStatusColor("#0f0");
    } else if (isOpen) {
      if (currentStateRef.current === STATE.TREE || currentStateRef.current === STATE.ORBIT) {
        currentStateRef.current = STATE.SCATTER;
        lastHandPosRef.current.x = handPosRef.current.x;
        lastHandPosRef.current.y = handPosRef.current.y;
      }
      setStatus("Open Hand: Nebula Scattered");
      setStatusColor("#00aaff");
    }

    wasPinchRef.current = isPinch;
  }, []);

  const drawSkeleton = (ctx: CanvasRenderingContext2D, landmarks: any[], w: number, h: number) => {
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00ff88';
    ctx.fillStyle = '#ff0044';
    const connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

    ctx.beginPath();
    for (const c of connections) {
      const s = landmarks[c[0]];
      const e = landmarks[c[1]];
      ctx.moveTo(s.x * w, s.y * h);
      ctx.lineTo(e.x * w, e.y * h);
    }
    ctx.stroke();
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  // --- Three.js Setup & Loops ---
  const randomSpherePoint = (r: number) => {
    const u = Math.random(), v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    return new THREE.Vector3(
      r * Math.sin(phi) * Math.cos(theta),
      r * Math.sin(phi) * Math.sin(theta),
      r * Math.cos(phi)
    );
  };

  const createInstancedMesh = (geo: THREE.BufferGeometry, mat: THREE.Material, count: number, dataArray: ParticleData[]) => {
    const mesh = new THREE.InstancedMesh(geo, mat, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mainGroupRef.current.add(mesh);

    for (let i = 0; i < count; i++) {
      const h = Math.random() * CONFIG.treeHeight - CONFIG.treeHeight / 2;
      const normH = (h + CONFIG.treeHeight / 2) / CONFIG.treeHeight;
      const rMax = CONFIG.maxRadius * (1 - normH) * 1.1;

      const r = Math.sqrt(Math.random()) * rMax;
      const theta = Math.random() * Math.PI * 2;

      const treePos = new THREE.Vector3(r * Math.cos(theta), h, r * Math.sin(theta));
      const scatterPos = randomSpherePoint(40 + Math.random() * 50);

      dataArray.push({
        treePos: treePos,
        scatterPos: scatterPos,
        currentPos: treePos.clone(),
        scale: 0.5 + Math.random() * 0.9,
        rotSpeed: new THREE.Euler(Math.random() * 0.02, Math.random() * 0.02, 0),
        rotation: new THREE.Euler(Math.random() * Math.PI, Math.random() * Math.PI, 0)
      });
    }
    return mesh;
  };

  const addPhotoMesh = useCallback((img: HTMLImageElement) => {
    const tex = new THREE.Texture(img);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;

    // Standard Polaroid Dimensions
    let w = 4, h = 4;
    if (img.width > img.height) h = 4 * (img.height / img.width);
    else w = 4 * (img.width / img.height);

    const geo = new THREE.PlaneGeometry(w, h);
    
    // Significantly Lower Brightness for the Polaroid Content
    const mat = new THREE.MeshStandardMaterial({
      map: tex, 
      side: THREE.DoubleSide,
      color: 0x888888, // Desaturated and dimmed
      emissive: 0x111111, // Very low ambient glow
      emissiveIntensity: 0.05, 
      roughness: 0.4, // Matte photo finish
      metalness: 0.0
    });
    
    const mesh = new THREE.Mesh(geo, mat);

    // Authentic Polaroid Frame (White Border)
    const frameWidth = w + 0.4;
    const frameHeight = h + 1.2;
    const frameGeo = new THREE.BoxGeometry(frameWidth, frameHeight, 0.08);
    const frameMat = new THREE.MeshStandardMaterial({ 
      color: 0xefefef, // Off-white paper
      roughness: 0.9,
      metalness: 0.0
    });
    const frame = new THREE.Mesh(frameGeo, frameMat);
    
    // Position frame so the bottom border is thick (Classic Polaroid)
    const bottomGap = (1.2 - 0.4) / 2;
    frame.position.z = -0.05;
    frame.position.y = -bottomGap;
    
    mesh.add(frame);

    const h_pos = (Math.random() - 0.5) * CONFIG.treeHeight;
    const normH = (h_pos + CONFIG.treeHeight / 2) / CONFIG.treeHeight;
    const maxR = CONFIG.maxRadius * (1 - normH) + 2;

    const r = maxR * (0.6 + 0.4 * Math.random());
    const theta = Math.random() * Math.PI * 2;

    const treePos = new THREE.Vector3(r * Math.cos(theta), h_pos, r * Math.sin(theta));
    const scatterPos = randomSpherePoint(60);

    mesh.userData = {
      treePos,
      scatterPos,
      baseRot: new THREE.Euler(0, theta + Math.PI / 2, 0)
    };
    mesh.position.copy(treePos);
    mesh.lookAt(new THREE.Vector3(0, h_pos, 0));

    photoMeshesRef.current.push(mesh);
    mainGroupRef.current.add(mesh);
  }, []);

  const updatePhotoLogic = (dummy: THREE.Object3D) => {
    const totalPhotos = photoMeshesRef.current.length;
    const photosPerLayer = 8;
    const layerCount = Math.ceil(totalPhotos / photosPerLayer);
    const camera = cameraRef.current;
    if (!camera) return;

    photoMeshesRef.current.forEach((mesh, idx) => {
      // Material logic for Lower Brightness
      const mat = (mesh.material as THREE.MeshStandardMaterial);
      if (currentStateRef.current === STATE.ZOOM && idx === zoomTargetIndexRef.current) {
        mat.color.lerp(COLOR_FULL, 0.1);
        mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, 0.3, 0.1); // Capped for low brightness
      } else {
        mat.color.lerp(COLOR_DIM, 0.1);
        mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, 0.02, 0.1); // Dimmed out
      }

      let targetPos, targetScale = 2.0;
      let lookTarget = camera.position;

      if (currentStateRef.current === STATE.ZOOM && idx === zoomTargetIndexRef.current) {
        const cameraDir = new THREE.Vector3();
        camera.getWorldDirection(cameraDir);
        const targetWorldPos = camera.position.clone().add(cameraDir.multiplyScalar(40));
        targetPos = mainGroupRef.current.worldToLocal(targetWorldPos);
        targetScale = 5.0;
      } else if (currentStateRef.current === STATE.ORBIT) {
        const layerIndex = Math.floor(idx / photosPerLayer);
        const idxInLayer = idx % photosPerLayer;
        const layerHeightStep = CONFIG.treeHeight / (layerCount + 1);
        const y = -CONFIG.treeHeight / 2 + (layerIndex + 1) * layerHeightStep;
        const normH = (y + CONFIG.treeHeight / 2) / CONFIG.treeHeight;
        const radius = (CONFIG.maxRadius * (1 - normH) * 1.5) + 10;
        const theta = (idxInLayer / Math.min(photosPerLayer, totalPhotos - layerIndex * photosPerLayer)) * Math.PI * 2;

        targetPos = new THREE.Vector3(radius * Math.cos(theta), y, radius * Math.sin(theta));
        targetScale = 3.5;

        const localCenter = new THREE.Vector3(0, targetPos.y, 0);
        const currentLocalDir = new THREE.Vector3().subVectors(mesh.position, localCenter).normalize();
        const localLookTarget = mesh.position.clone().add(currentLocalDir);
        lookTarget = localLookTarget.applyMatrix4(mainGroupRef.current.matrixWorld);
      } else {
        targetPos = currentStateRef.current === STATE.TREE ? mesh.userData.treePos : mesh.userData.scatterPos;
        if (currentStateRef.current !== STATE.TREE) mesh.position.y += Math.sin(timeRef.current + idx) * 0.02;
        if (currentStateRef.current === STATE.SCATTER) targetScale = 3.0;

        if (currentStateRef.current === STATE.TREE) {
          mesh.rotation.copy(mesh.userData.baseRot);
          mesh.rotation.y += 0.01;
        }
      }

      mesh.position.lerp(targetPos, 0.08);
      const currentScale = mesh.scale.x;
      const newScale = THREE.MathUtils.lerp(currentScale, targetScale, 0.1);
      mesh.scale.setScalar(newScale);

      if (currentStateRef.current === STATE.ORBIT) {
        mesh.lookAt(lookTarget);
      } else if (currentStateRef.current !== STATE.TREE) {
        mesh.lookAt(camera.position);
      }
    });
  };

  const animate = useCallback(() => {
    const dummy = new THREE.Object3D();
    const updateMeshLogic = (mesh: THREE.InstancedMesh | null, dataArray: ParticleData[]) => {
      if (!mesh) return;
      const isZoom = currentStateRef.current === STATE.ZOOM;
      const isTree = currentStateRef.current === STATE.TREE;
      const isOrbit = currentStateRef.current === STATE.ORBIT;

      for (let i = 0; i < dataArray.length; i++) {
        const item = dataArray[i];
        let target;
        if (isTree || isOrbit) target = item.treePos;
        else target = item.scatterPos;
        if (isZoom) target = item.scatterPos;

        if (!isTree && !isOrbit) item.currentPos.y += Math.sin(timeRef.current + i * 0.1) * 0.01;
        item.currentPos.lerp(target, 0.08);
        item.rotation.x += item.rotSpeed.x;
        item.rotation.y += item.rotSpeed.y;

        let s = item.scale;
        if (isZoom) s *= 0.5;

        dummy.position.copy(item.currentPos);
        dummy.rotation.copy(item.rotation);
        dummy.scale.setScalar(s);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
    };

    const loop = () => {
      requestAnimationFrame(loop);
      timeRef.current += 0.01;

      updateMeshLogic(meshesRef.current.gold, logicDataRef.current.gold);
      updateMeshLogic(meshesRef.current.silver, logicDataRef.current.silver);
      updateMeshLogic(meshesRef.current.gem, logicDataRef.current.gem);
      updateMeshLogic(meshesRef.current.emerald, logicDataRef.current.emerald);

      // Dust
      if (meshesRef.current.dust) {
        const positions = meshesRef.current.dust.geometry.attributes.position.array as Float32Array;
        const isTreeForm = (currentStateRef.current === STATE.TREE || currentStateRef.current === STATE.ORBIT);
        for (let i = 0; i < logicDataRef.current.dust.length; i++) {
          const item = logicDataRef.current.dust[i];
          if (isTreeForm) {
            item.currentPos.y += item.velocity;
            if (item.currentPos.y > CONFIG.treeHeight / 2) item.currentPos.y = -CONFIG.treeHeight / 2;
            const rMax = CONFIG.maxRadius * (1 - (item.currentPos.y + CONFIG.treeHeight / 2) / CONFIG.treeHeight) + 5;
            const rCurr = Math.sqrt(item.currentPos.x ** 2 + item.currentPos.z ** 2);
            if (rCurr > rMax) {
              item.currentPos.x *= 0.95;
              item.currentPos.z *= 0.95;
            }
          } else {
            item.currentPos.lerp(item.scatterPos, 0.05);
          }
          positions[i * 3] = item.currentPos.x;
          positions[i * 3 + 1] = item.currentPos.y;
          positions[i * 3 + 2] = item.currentPos.z;
        }
        meshesRef.current.dust.geometry.attributes.position.needsUpdate = true;
      }

      // Physics Rotation
      if (currentStateRef.current === STATE.ZOOM) {
        rotationVelocityRef.current.x *= 0.8;
        rotationVelocityRef.current.y *= 0.8;
      } else if (currentStateRef.current === STATE.ORBIT) {
        rotationVelocityRef.current.y = 0.005;
        rotationVelocityRef.current.x *= 0.9;
        mainGroupRef.current.rotation.y += rotationVelocityRef.current.y;
        mainGroupRef.current.rotation.x = THREE.MathUtils.lerp(mainGroupRef.current.rotation.x, 0, 0.05);
      } else if (currentStateRef.current === STATE.SCATTER) {
        if (isHandPresentRef.current) {
          const deltaX = handPosRef.current.x - lastHandPosRef.current.x;
          const deltaY = handPosRef.current.y - lastHandPosRef.current.y;
          if (Math.abs(deltaX) > 0.001) rotationVelocityRef.current.y += deltaX * 0.2;
          if (Math.abs(deltaY) > 0.001) rotationVelocityRef.current.x += deltaY * 0.1;
          lastHandPosRef.current.x = handPosRef.current.x;
          lastHandPosRef.current.y = handPosRef.current.y;
        }
        mainGroupRef.current.rotation.y += rotationVelocityRef.current.y;
        mainGroupRef.current.rotation.x += rotationVelocityRef.current.x;
        rotationVelocityRef.current.y *= 0.95;
        rotationVelocityRef.current.x *= 0.95;
        mainGroupRef.current.rotation.x *= 0.98;
      } else {
        mainGroupRef.current.rotation.y += 0.004;
        mainGroupRef.current.rotation.x *= 0.95;
      }

      mainGroupRef.current.updateMatrixWorld();
      updatePhotoLogic(dummy);

      if (logicDataRef.current.star) {
        const star = logicDataRef.current.star;
        const target = (currentStateRef.current === STATE.TREE || currentStateRef.current === STATE.ORBIT)
          ? star.userData.treePos
          : star.userData.scatterPos;
        star.position.lerp(target, 0.05);
        star.rotation.y += 0.01;
      }

      if (composerRef.current) composerRef.current.render();
    };
    loop();
  }, []);

  useEffect(() => {
    if (!canvasRef.current) return;

    const width = canvasRef.current.clientWidth;
    const height = canvasRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505);
    scene.fog = new THREE.FogExp2(0x050505, 0.002);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 0, CONFIG.camDistance);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    canvasRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;

    scene.add(new THREE.AmbientLight(0xffffff, 0.2));
    const spotLight = new THREE.SpotLight(0xffddaa, 100);
    spotLight.position.set(30, 80, 50);
    spotLight.angle = Math.PI / 4;
    spotLight.penumbra = 1;
    scene.add(spotLight);
    const blueLight = new THREE.PointLight(0x4488ff, 50, 200);
    blueLight.position.set(-40, -20, 40);
    scene.add(blueLight);

    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(width, height), 1.5, 0.4, 0.85);
    bloomPass.threshold = CONFIG.bloomThreshold;
    bloomPass.strength = CONFIG.bloomStrength;
    bloomPass.radius = CONFIG.bloomRadius;

    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Create particles
    const goldMat = new THREE.MeshPhysicalMaterial({ color: 0xffaa00, metalness: 1.0, roughness: 0.15, clearcoat: 1.0, emissive: 0xaa5500, emissiveIntensity: 0.1 });
    const silverMat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.9, roughness: 0.2, clearcoat: 1.0 });
    const gemMat = new THREE.MeshPhysicalMaterial({ color: 0xff0044, metalness: 0.1, roughness: 0.0, transmission: 0.6, thickness: 1.5, emissive: 0x550011, emissiveIntensity: 0.4 });
    const emeraldMat = new THREE.MeshPhysicalMaterial({ color: 0x00aa55, metalness: 0.2, roughness: 0.1, transmission: 0.5, thickness: 1.5, emissive: 0x002211, emissiveIntensity: 0.3 });

    meshesRef.current.gold = createInstancedMesh(new THREE.SphereGeometry(0.7, 12, 12), goldMat, CONFIG.goldCount, logicDataRef.current.gold);
    meshesRef.current.silver = createInstancedMesh(new THREE.BoxGeometry(0.8, 0.8, 0.8), silverMat, CONFIG.silverCount, logicDataRef.current.silver);
    meshesRef.current.gem = createInstancedMesh(new THREE.OctahedronGeometry(0.8, 0), gemMat, CONFIG.gemCount, logicDataRef.current.gem);
    meshesRef.current.emerald = createInstancedMesh(new THREE.ConeGeometry(0.5, 1.2, 6), emeraldMat, CONFIG.emeraldCount, logicDataRef.current.emerald);

    // Star
    const starGeo = new THREE.OctahedronGeometry(3.0, 0);
    const starMat = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.8, roughness: 0, emissive: 0xffffee, emissiveIntensity: 1.5 });
    const star = new THREE.Mesh(starGeo, starMat);
    star.userData = { treePos: new THREE.Vector3(0, CONFIG.treeHeight / 2 + 2.5, 0), scatterPos: new THREE.Vector3(0, 60, 0) };
    star.position.copy(star.userData.treePos);
    mainGroupRef.current.add(star);
    logicDataRef.current.star = star;

    // Dust
    const dustGeo = new THREE.BufferGeometry();
    const dustPos = new Float32Array(CONFIG.dustCount * 3);
    for (let i = 0; i < CONFIG.dustCount; i++) {
      const h = Math.random() * CONFIG.treeHeight - CONFIG.treeHeight / 2;
      const r = Math.random() * CONFIG.maxRadius + 5;
      const theta = Math.random() * Math.PI * 2;
      const x = r * Math.cos(theta), z = r * Math.sin(theta);
      dustPos[i * 3] = x; dustPos[i * 3 + 1] = h; dustPos[i * 3 + 2] = z;
      logicDataRef.current.dust.push({
        treePos: new THREE.Vector3(x, h, z), scatterPos: randomSpherePoint(70), currentPos: new THREE.Vector3(x, h, z), velocity: Math.random() * 0.05 + 0.02
      });
    }
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
    const dustMat = new THREE.PointsMaterial({ color: 0xffd700, size: 0.5, transparent: true, opacity: 0.4, blending: THREE.AdditiveBlending, depthWrite: false });
    meshesRef.current.dust = new THREE.Points(dustGeo, dustMat);
    mainGroupRef.current.add(meshesRef.current.dust);

    // StarField
    const starFieldGeo = new THREE.BufferGeometry();
    const starFieldPos = [];
    for (let i = 0; i < 800; i++) starFieldPos.push((Math.random() - 0.5) * 800, (Math.random() - 0.5) * 800, (Math.random() - 0.5) * 800);
    starFieldGeo.setAttribute('position', new THREE.Float32BufferAttribute(starFieldPos, 3));
    scene.add(new THREE.Points(starFieldGeo, new THREE.PointsMaterial({ color: 0x666666, size: 1.0, transparent: true, opacity: 0.6 })));

    // Bg Particles
    const bgGeo = new THREE.BufferGeometry();
    const bgPos = [];
    for (let i = 0; i < 1200; i++) bgPos.push((Math.random() - 0.5) * 600, (Math.random() - 0.5) * 600, (Math.random() - 0.5) * 600);
    bgGeo.setAttribute('position', new THREE.Float32BufferAttribute(bgPos, 3));
    meshesRef.current.bgParticles = new THREE.Points(bgGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 1.2, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending }));
    scene.add(meshesRef.current.bgParticles);

    scene.add(mainGroupRef.current);

    // Start Loops
    animate();

    const handleResize = () => {
      if (!canvasRef.current || !cameraRef.current || !rendererRef.current || !composerRef.current) return;
      const w = canvasRef.current.clientWidth;
      const h = canvasRef.current.clientHeight;
      cameraRef.current.aspect = w / h;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(w, h);
      composerRef.current.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Init MediaPipe
    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO", numHands: 1
        });

        if (videoRef.current) {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true });
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener("loadeddata", () => {
            setIsLoading(false);
            setStatus("Engine Ready. Reach out your hand!");
            setStatusColor("#0f0");
            predictLoop(landmarker);
          });
        }
      } catch (e) {
        console.error(e);
        setIsLoading(false);
        setStatus("Camera Access Denied");
        setStatusColor("#f00");
      }
    };

    let lastVideoTime = -1;
    const predictLoop = (landmarker: HandLandmarker) => {
      if (videoRef.current && videoRef.current.currentTime !== lastVideoTime) {
        lastVideoTime = videoRef.current.currentTime;
        const result = landmarker.detectForVideo(videoRef.current, performance.now());
        processPrediction(result);
      }
      requestAnimationFrame(() => predictLoop(landmarker));
    };

    const processPrediction = (result: HandLandmarkerResult) => {
      const canvas = skeletonCanvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (result.landmarks && result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        isHandPresentRef.current = true;
        drawSkeleton(ctx, landmarks, canvas.width, canvas.height);
        handleGesture(landmarks);
      } else {
        isHandPresentRef.current = false;
        wasPinchRef.current = false;
      }
    };

    initMediaPipe();

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [animate, handleGesture]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    for (let i = 0; i < files.length; i++) {
      const reader = new FileReader();
      reader.onload = (evt) => {
        const img = new Image();
        img.src = evt.target?.result as string;
        img.onload = () => addPhotoMesh(img);
      };
      reader.readAsDataURL(files[i]);
    }
    e.target.value = '';
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  };

  return (
    <div
      ref={containerRef}
      className="bg-black flex items-center justify-center overflow-hidden w-full h-full relative"
    >
      <div className="aspect-4-3-container bg-neutral-950 overflow-hidden shadow-2xl relative" ref={canvasRef}>
        {/* UI Overlay */}
        <div className="absolute top-8 left-0 w-full flex flex-col items-center pointer-events-none z-50">
          <h1
            className="text-4xl font-light tracking-[0.5rem] uppercase cursor-pointer pointer-events-auto transition-transform hover:scale-105 active:scale-95"
            style={{
              background: 'linear-gradient(to right, #fff, #ffd700, #fff)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 0 10px rgba(212, 175, 55, 0.5))'
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            Merry Christmas to 7
          </h1>
          <p className="text-[10px] text-white/40 tracking-widest mt-2 opacity-0 transition-opacity hover:opacity-100 group-hover:opacity-100">
            CLICK TITLE TO UPLOAD MEMORIES
          </p>
        </div>

        {/* Status and Instructions */}
        <div className="absolute top-8 left-8 z-30 pointer-events-none flex flex-col gap-4">
          <div className="flex items-center gap-2 text-sm font-light">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: statusColor }}></span>
            <span className="text-white/80">{status}</span>
          </div>
          <div className="text-[10px] text-white/50 leading-relaxed max-w-[200px] p-4 rounded-lg bg-black/20 backdrop-blur-sm border-l border-yellow-500/30">
            • <span className="text-yellow-500/80">FIST</span> : Aggregate Tree<br/>
            • <span className="text-yellow-500/80">OPEN</span> : Nebula Scatter<br/>
            • <span className="text-yellow-500/80">PEACE</span> : Wish Tree Orbit<br/>
            • <span className="text-yellow-500/80">PINCH</span> : Next Photo Cycle<br/>
            • <span className="text-yellow-500/80">WAVE</span> : Rotate View (Scatter)
          </div>
        </div>

        {/* Video Feedback (Mirrored) */}
        <div className="absolute top-5 right-5 w-40 h-32 z-20 border border-white/20 rounded-xl overflow-hidden shadow-2xl bg-black/80 opacity-40 hover:opacity-100 transition-opacity pointer-events-none scale-x-[-1]">
          <video ref={videoRef} className="w-full h-full object-cover opacity-50 absolute inset-0" autoPlay playsInline muted />
          <canvas ref={skeletonCanvasRef} className="absolute inset-0 w-full h-full z-10" />
        </div>

        {/* Fullscreen Button */}
        <button
          onClick={toggleFullscreen}
          className="absolute bottom-8 right-8 z-40 bg-black/30 text-yellow-500 border border-yellow-500/30 w-11 h-11 rounded-full flex items-center justify-center hover:bg-yellow-500 hover:text-black transition-all backdrop-blur-md"
          title="Toggle Fullscreen"
        >
          ⛶
        </button>

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 z-[100] flex items-center justify-center bg-black/90 text-yellow-500 tracking-[0.3em] font-light animate-pulse">
            CRAFTING MAGIC...
          </div>
        )}

        <input
          type="file"
          ref={fileInputRef}
          multiple
          accept="image/*"
          className="hidden"
          onChange={onFileChange}
        />
      </div>
    </div>
  );
};

export default App;
