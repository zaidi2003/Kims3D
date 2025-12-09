import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Overlay
const overlay = document.createElement('div');
overlay.style.position = 'fixed';
overlay.style.top = '0';
overlay.style.left = '0';
overlay.style.width = '100vw';
overlay.style.height = '100vh';
overlay.style.backgroundColor = '#202020';
overlay.style.display = 'flex';
overlay.style.flexDirection = 'column';
overlay.style.justifyContent = 'center';
overlay.style.alignItems = 'center';
overlay.style.color = 'white';
overlay.style.fontSize = '24px';
overlay.style.fontFamily = 'Arial, Helvetica, sans-serif';
overlay.style.zIndex = '100';
overlay.innerText = 'KIMS Bookstore Virtual Experience';

const enterBtn = document.createElement('button');
enterBtn.innerText = 'Enter Virtual Experience';
enterBtn.style.marginTop = '20px';
enterBtn.style.padding = '12px 24px';
enterBtn.style.fontSize = '18px';
enterBtn.style.borderRadius = '8px';
enterBtn.style.border = 'none';
enterBtn.style.backgroundColor = '#4caf50';
enterBtn.style.color = 'white';
enterBtn.style.cursor = 'pointer';

overlay.appendChild(enterBtn);
document.body.appendChild(overlay);

enterBtn.addEventListener('click', () => {
  overlay.style.display = 'none';
});


// Scene, Camera, Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202020);

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 1));
const light = new THREE.DirectionalLight(0xffffff, 2);
light.position.set(1, 1, 1);
scene.add(light);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;


// Hotspots Gotten From XML
const hotspots = [
  { position: new THREE.Vector3(-0.52068, 0.114658, 0.54195), image: '25.jpg' },
  { position: new THREE.Vector3(-0.54345, 0.113501, 0.47590), image: '24.jpg' },
  { position: new THREE.Vector3(-0.53575, 0.11133, 0.38265), image: '23.jpg' },
  { position: new THREE.Vector3(-0.53731, 0.11560, 0.35203), image: '22.jpg' },
  { position: new THREE.Vector3(-0.52767, 0.11237, 0.29128), image: '21.jpg' },
  { position: new THREE.Vector3(-0.51914, 0.11823, 0.19846), image: '20.jpg' },
  { position: new THREE.Vector3(-0.52525, 0.11577, 0.15497), image: '19.jpg' },
  { position: new THREE.Vector3(-0.47442, 0.11859, 0.09271), image: '18.jpg' },
  { position: new THREE.Vector3(-0.43687, 0.11824, 0.05498), image: '17.jpg' },
  { position: new THREE.Vector3(-0.39054, 0.11751, 0.01350), image: '16.jpg' },
  { position: new THREE.Vector3(-0.36604, 0.11272, -0.00411), image: '15.jpg' },
  { position: new THREE.Vector3(-0.33614, 0.11766, -0.03012), image: '14.jpg' },
  { position: new THREE.Vector3(-0.29544, 0.11172, -0.05347), image: '13.jpg' },
  { position: new THREE.Vector3(-0.21957, 0.11284, -0.07906), image: '12.jpg' },
  { position: new THREE.Vector3(-0.16526, 0.11838, -0.09286), image: '11.jpg' },
  { position: new THREE.Vector3(-0.07591, 0.11366, -0.09905), image: '10.jpg' },
  { position: new THREE.Vector3(-0.05792, 0.11307, -0.09755), image: '9.jpg' },
  { position: new THREE.Vector3(-0.04015, 0.11266, -0.09583), image: '8.jpg' },
  { position: new THREE.Vector3(0.00022, 0.11331, -0.09227), image: '7.jpg' },
  { position: new THREE.Vector3(0.06569, 0.11360, -0.08891), image: '6.jpg' },
  { position: new THREE.Vector3(0.13698, 0.11761, -0.08701), image: '5.jpg' },
  { position: new THREE.Vector3(0.19065, 0.11180, -0.07151), image: '4.jpg' },
  { position: new THREE.Vector3(0.24500, 0.11731, -0.03961), image: '3.jpg' },
  { position: new THREE.Vector3(0.28384, 0.12240, -0.00694), image: '2.jpg' },
  { position: new THREE.Vector3(0.30845, 0.12628, 0.01315), image: '1.jpg' },
];

const hotspotGroup = new THREE.Group();
scene.add(hotspotGroup);


// Load Model
let currentModel = null;

const loader = new GLTFLoader();
loader.load(
  '/model.glb',
  (gltf) => {
    currentModel = gltf.scene;

    const box = new THREE.Box3().setFromObject(currentModel);
    const size = box.getSize(new THREE.Vector3());
    const scale = 2 / Math.max(size.x, size.y, size.z);
    currentModel.scale.setScalar(scale);

    scene.add(currentModel);

    // show predefined hotspots (like google street view)
    hotspots.forEach((spot) => {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.03, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0xff0000 })
      );
      sphere.position.copy(spot.position);
      sphere.userData.image = spot.image;
      hotspotGroup.add(sphere);
    });
  },
  undefined,
  (err) => {
    console.error('Error loading model:', err);
  }
);

// Popup Image Viewer
const imgPopup = document.createElement('div');
imgPopup.style.position = 'fixed';
imgPopup.style.top = '0';
imgPopup.style.left = '0';
imgPopup.style.width = '100vw';
imgPopup.style.height = '100vh';
imgPopup.style.background = 'rgba(0,0,0,0.8)';
imgPopup.style.display = 'none';
imgPopup.style.justifyContent = 'center';
imgPopup.style.alignItems = 'center';
imgPopup.style.zIndex = '200';

const imgElement = document.createElement('img');
imgElement.style.maxWidth = '80%';
imgElement.style.maxHeight = '80%';


imgPopup.appendChild(imgElement);
document.body.appendChild(imgPopup);
imgPopup.addEventListener('click', () => (imgPopup.style.display = 'none'));

function openImagePopup(src) {
  imgElement.src = src;
  imgPopup.style.display = 'flex';
}


const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

window.addEventListener('click', (event) => {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(hotspotGroup.children);

  if (intersects.length > 0) {
    const clicked = intersects[0].object;
    if (clicked.userData.image) openImagePopup(clicked.userData.image);
  }
});

// ---------------------
// Animate
// ---------------------
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// Resize + Zoom Controls
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const zoomContainer = document.createElement('div');
zoomContainer.style.position = 'absolute';
zoomContainer.style.bottom = '40px';
zoomContainer.style.right = '40px';
zoomContainer.style.display = 'flex';
zoomContainer.style.flexDirection = 'column';
zoomContainer.style.gap = '10px';
document.body.appendChild(zoomContainer);

const zoomInBtn = document.createElement('button');
zoomInBtn.innerText = '+';
const zoomOutBtn = document.createElement('button');
zoomOutBtn.innerText = '-';

[zoomInBtn, zoomOutBtn].forEach((btn) => {
  btn.style.padding = '12px 16px';
  btn.style.fontSize = '20px';
  btn.style.borderRadius = '8px';
  btn.style.border = 'none';
  btn.style.cursor = 'pointer';
  btn.style.backgroundColor = '#4caf50';
  btn.style.color = 'white';
  zoomContainer.appendChild(btn);
});

const zoomStep = 0.2;
zoomInBtn.addEventListener('click', () => {
  camera.position.multiplyScalar(1 - zoomStep);
});
zoomOutBtn.addEventListener('click', () => {
  camera.position.multiplyScalar(1 + zoomStep);
});


// Toggle Hotspots Button, so that one can view model without distractions
// Initially hide hotspots
hotspotGroup.visible = false;

const toggleHotspotsBtn = document.createElement('button');
toggleHotspotsBtn.innerText = 'Show Hotspots'; // initially show button text as "Show Hotspots"
toggleHotspotsBtn.style.padding = '12px 16px';
toggleHotspotsBtn.style.fontSize = '16px';
toggleHotspotsBtn.style.borderRadius = '8px';
toggleHotspotsBtn.style.border = 'none';
toggleHotspotsBtn.style.cursor = 'pointer';
toggleHotspotsBtn.style.backgroundColor = '#f44336'; // red
toggleHotspotsBtn.style.color = 'white';

// add below zoom buttons
zoomContainer.appendChild(toggleHotspotsBtn);

let hotspotsVisible = false;

toggleHotspotsBtn.addEventListener('click', () => {
  hotspotsVisible = !hotspotsVisible;
  hotspotGroup.visible = hotspotsVisible;
  toggleHotspotsBtn.innerText = hotspotsVisible ? 'Hide Hotspots' : 'Show Hotspots';
});





// Advanced Controls (Left Panel)
const advancedControls = document.createElement('div');
advancedControls.style.position = 'absolute';
advancedControls.style.bottom = '20px';
advancedControls.style.left = '20px';
advancedControls.style.transform = 'none';
advancedControls.style.display = 'none'; // hidden initially
advancedControls.style.flexDirection = 'column';
advancedControls.style.gap = '10px';
advancedControls.style.background = 'rgba(0,0,0,0.5)';
advancedControls.style.padding = '10px';
advancedControls.style.borderRadius = '8px';
advancedControls.style.zIndex = '300';
document.body.appendChild(advancedControls);

// X rotation slider
const xLabel = document.createElement('label');
xLabel.innerText = 'Rotate X';
xLabel.style.color = 'white';
const xSlider = document.createElement('input');
xSlider.type = 'range';
xSlider.min = '-180';
xSlider.max = '180';
xSlider.value = '0';
xSlider.step = '1';

advancedControls.appendChild(xLabel);
advancedControls.appendChild(xSlider);

// Y rotation slider
const yLabel = document.createElement('label');
yLabel.innerText = 'Rotate Y';
yLabel.style.color = 'white';
const ySlider = document.createElement('input');
ySlider.type = 'range';
ySlider.min = '-180';
ySlider.max = '180';
ySlider.value = '0';
ySlider.step = '1';

advancedControls.appendChild(yLabel);
advancedControls.appendChild(ySlider);

// Slider Events
xSlider.addEventListener('input', () => {
  if (currentModel) {
    currentModel.rotation.x = THREE.MathUtils.degToRad(parseFloat(xSlider.value));
  }
});

ySlider.addEventListener('input', () => {
  if (currentModel) {
    currentModel.rotation.y = THREE.MathUtils.degToRad(parseFloat(ySlider.value));
  }
});

// Show advanced controls when entering experience
enterBtn.addEventListener('click', () => {
  overlay.style.display = 'none';
  advancedControls.style.display = 'flex'; // show controls now
});

// green colored sliders
[xSlider, ySlider].forEach((slider) => {
  slider.style.width = '120px';
  slider.style.background = '#4caf50'; 
  slider.style.accentColor = '#4caf50'; 
  slider.style.cursor = 'pointer';
});
