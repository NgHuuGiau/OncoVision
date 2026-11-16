// OrbitControls (standalone, non-ES-module)
// Attach to window.THREE.OrbitControls so 3d-viewer.js can use it directly.
// Chức năng cơ bản: xoay (chuột trái), zoom (cuộn chuột), di chuyển (chuột phải).
(function (root) {
  'use strict';

  var THREE = root.THREE;
  if (!THREE) {
    console.error('OrbitControls: THREE chưa được load');
    return;
  }

  var MOUSE = { LEFT: 0, MIDDLE: 1, RIGHT: 2 };
  var ROTATE = 0, DOLLY = 1, PAN = 2;

  var _changeEvent = { type: 'change' };
  var _startEvent = { type: 'start' };
  var _endEvent = { type: 'end' };

  var DEG2RAD = Math.PI / 180;

  var OrbitControls = function (object, domElement) {
    if (domElement === undefined) console.warn('OrbitControls: Không có phần tử DOM');

    this.object = object;
    this.domElement = domElement;

    this.enabled = true;
    this.target = new THREE.Vector3();
    this.minDistance = 0;
    this.maxDistance = Infinity;
    this.minZoom = 0;
    this.maxZoom = Infinity;
    this.minPolarAngle = 0;
    this.maxPolarAngle = Math.PI;
    this.minAzimuthAngle = -Infinity;
    this.maxAzimuthAngle = Infinity;
    this.enableDamping = false;
    this.dampingFactor = 0.05;
    this.enableZoom = true;
    this.zoomSpeed = 1.0;
    this.enableRotate = true;
    this.rotateSpeed = 1.0;
    this.enablePan = true;
    this.panSpeed = 1.0;
    this.screenSpacePanning = true;
    this.keyPanSpeed = 7.0;
    this.autoRotate = false;
    this.autoRotateSpeed = 2.0;

    this.target0 = this.target.clone();
    this.position0 = this.object.position.clone();
    this.zoom0 = this.object.zoom;

    var scope = this;
    var state = -1;
    var EPS = 0.000001;

    var spherical = new THREE.Spherical();
    var sphericalDelta = new THREE.Spherical();
    var scale = 1;
    var panOffset = new THREE.Vector3();

    var rotateStart = new THREE.Vector2();
    var rotateEnd = new THREE.Vector2();
    var rotateDelta = new THREE.Vector2();

    var panStart = new THREE.Vector2();
    var panEnd = new THREE.Vector2();
    var panDelta = new THREE.Vector2();

    var dollyStart = new THREE.Vector2();
    var dollyEnd = new THREE.Vector2();
    var dollyDelta = new THREE.Vector2();

    var pointers = [];
    var pointerPositions = {};

    // --- API ---

    this.getPolarAngle = function () { return spherical.phi; };
    this.getAzimuthalAngle = function () { return spherical.theta; };

    this.listenToKeyEvents = function (domElement) {
      domElement.addEventListener('keydown', onKeyDown);
    };
    this.stopListenKeyEvents = function (domElement) {
      domElement.removeEventListener('keydown', onKeyDown);
    };

    this.saveState = function () {
      scope.target0.copy(scope.target);
      scope.position0.copy(scope.object.position);
      scope.zoom0 = scope.object.zoom;
    };

    this.reset = function () {
      scope.target.copy(scope.target0);
      scope.object.position.copy(scope.position0);
      scope.object.zoom = scope.zoom0;
      scope.object.updateProjectionMatrix();
      scope.dispatchEvent(_changeEvent);
      scope.update();
      state = -1;
    };

    this.update = function () {
      var offset = new THREE.Vector3();
      var quat = new THREE.Quaternion().setFromUnitVectors(object.up, new THREE.Vector3(0, 1, 0));
      var quatInverse = quat.clone().invert();
      var lastPosition = new THREE.Vector3();
      var lastQuaternion = new THREE.Quaternion();
      var lastTargetPosition = new THREE.Vector3();

      return function update() {
        var position = scope.object.position;
        offset.copy(position).sub(scope.target);
        offset.applyQuaternion(quat);
        spherical.setFromVector3(offset);

        if (scope.autoRotate && state === -1) {
          rotateLeft(2 * Math.PI / 60 / 60 * scope.autoRotateSpeed);
        }

        if (scope.enableDamping) {
          spherical.theta += sphericalDelta.theta * scope.dampingFactor;
          spherical.phi += sphericalDelta.phi * scope.dampingFactor;
        } else {
          spherical.theta += sphericalDelta.theta;
          spherical.phi += sphericalDelta.phi;
        }

        spherical.theta = Math.max(scope.minAzimuthAngle, Math.min(scope.maxAzimuthAngle, spherical.theta));
        spherical.phi = Math.max(scope.minPolarAngle, Math.min(scope.maxPolarAngle, spherical.phi));
        spherical.makeSafe();

        spherical.radius = scale * spherical.radius;
        spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));
        scale = 1;

        offset.setFromSpherical(spherical);
        offset.applyQuaternion(quatInverse);
        position.copy(scope.target).add(offset);
        scope.object.lookAt(scope.target);

        if (scope.enableDamping) {
          sphericalDelta.theta *= (1 - scope.dampingFactor);
          sphericalDelta.phi *= (1 - scope.dampingFactor);
          panOffset.multiplyScalar(1 - scope.dampingFactor);
        } else {
          sphericalDelta.set(0, 0, 0);
          panOffset.set(0, 0, 0);
        }

        var zoomChanged = false;
        if (scope.object.isPerspectiveCamera) {
          // nothing
        }

        if (
          zoomChanged ||
          lastPosition.distanceToSquared(scope.object.position) > EPS ||
          8 * (1 - lastQuaternion.dot(scope.object.quaternion)) > EPS ||
          lastTargetPosition.distanceToSquared(scope.target) > 0
        ) {
          scope.dispatchEvent(_changeEvent);
          lastPosition.copy(scope.object.position);
          lastQuaternion.copy(scope.object.quaternion);
          lastTargetPosition.copy(scope.target);
          return true;
        }
        return false;
      };
    }();

    this.dispose = function () {
      scope.domElement.removeEventListener('contextmenu', onContextMenu, false);
      scope.domElement.removeEventListener('pointerdown', onPointerDown, false);
      scope.domElement.removeEventListener('wheel', onMouseWheel, false);
      scope.domElement.removeEventListener('pointermove', onPointerMove, false);
      scope.domElement.removeEventListener('pointerup', onPointerUp, false);
    };

    // --- internals ---

    function rotateLeft(angle) {
      sphericalDelta.theta -= angle;
    }
    function rotateUp(angle) {
      sphericalDelta.phi -= angle;
    }
    function panLeft(distance, objectMatrix) {
      var v = new THREE.Vector3();
      v.setFromMatrixColumn(objectMatrix, 0);
      v.multiplyScalar(-distance);
      panOffset.add(v);
    }
    function panUp(distance, objectMatrix) {
      var v = new THREE.Vector3();
      if (scope.screenSpacePanning) {
        v.setFromMatrixColumn(objectMatrix, 1);
      } else {
        v.setFromMatrixColumn(objectMatrix, 0);
        v.crossVectors(scope.object.up, v);
      }
      v.multiplyScalar(distance);
      panOffset.add(v);
    }
    function pan(deltaX, deltaY) {
      var element = scope.domElement;
      var position = scope.object.position;
      var offset = new THREE.Vector3().copy(position).sub(scope.target);
      var targetDistance = offset.length();
      targetDistance *= Math.tan((scope.object.fov / 2) * DEG2RAD);
      panLeft(2 * deltaX * targetDistance / element.clientHeight, scope.object.matrix);
      panUp(2 * deltaY * targetDistance / element.clientHeight, scope.object.matrix);
    }
    function dollyIn(dollyScale) {
      if (scope.object.isPerspectiveCamera || scope.object.isOrthographicCamera) {
        scale /= dollyScale;
      }
    }
    function dollyOut(dollyScale) {
      if (scope.object.isPerspectiveCamera || scope.object.isOrthographicCamera) {
        scale *= dollyScale;
      }
    }
    function getZoomScale(delta) {
      return Math.pow(0.95, scope.zoomSpeed * delta / 100);
    }
    function updateZoomParameters(clientX, clientY) {} // no-op for simplicity

    function onPointerDown(event) {
      if (scope.enabled === false) return;
      if (pointers.length === 0) {
        scope.domElement.setPointerCapture(event.pointerId);
        scope.domElement.addEventListener('pointermove', onPointerMove, false);
        scope.domElement.addEventListener('pointerup', onPointerUp, false);
      }
      addPointer(event);
      if (event.pointerType === 'touch') {
        onTouchStart(event);
      } else {
        onMouseDown(event);
      }
    }
    function onPointerMove(event) {
      if (scope.enabled === false) return;
      if (event.pointerType === 'touch') {
        onTouchMove(event);
      } else {
        onMouseMove(event);
      }
    }
    function onPointerUp(event) {
      removePointer(event);
      if (pointers.length === 0) {
        scope.domElement.releasePointerCapture(event.pointerId);
        scope.domElement.removeEventListener('pointermove', onPointerMove, false);
        scope.domElement.removeEventListener('pointerup', onPointerUp, false);
      }
      scope.dispatchEvent(_endEvent);
      state = -1;
    }

    function onMouseDown(event) {
      var mouseAction;
      switch (event.button) {
        case 0: mouseAction = MOUSE.LEFT; break;
        case 1: mouseAction = MOUSE.MIDDLE; break;
        case 2: mouseAction = MOUSE.RIGHT; break;
        default: mouseAction = -1;
      }
      switch (mouseAction) {
        case MOUSE.DOLLY:
          if (scope.enableZoom === false) return;
          dollyStart.set(event.clientX, event.clientY);
          state = DOLLY;
          break;
        case MOUSE.ROTATE:
          if (event.ctrlKey || event.metaKey || event.shiftKey) {
            if (scope.enablePan === false) return;
            panStart.set(event.clientX, event.clientY);
            state = PAN;
          } else {
            if (scope.enableRotate === false) return;
            rotateStart.set(event.clientX, event.clientY);
            state = ROTATE;
          }
          break;
        case MOUSE.PAN:
          if (event.ctrlKey || event.metaKey || event.shiftKey) {
            if (scope.enableRotate === false) return;
            rotateStart.set(event.clientX, event.clientY);
            state = ROTATE;
          } else {
            if (scope.enablePan === false) return;
            panStart.set(event.clientX, event.clientY);
            state = PAN;
          }
          break;
        default: state = -1;
      }
      if (state !== -1) scope.dispatchEvent(_startEvent);
    }
    function onMouseMove(event) {
      switch (state) {
        case ROTATE:
          if (scope.enableRotate === false) return;
          rotateEnd.set(event.clientX, event.clientY);
          rotateDelta.subVectors(rotateEnd, rotateStart);
          var element = scope.domElement;
          rotateLeft(2 * Math.PI * rotateDelta.x / element.clientHeight * scope.rotateSpeed);
          rotateUp(2 * Math.PI * rotateDelta.y / element.clientHeight * scope.rotateSpeed);
          rotateStart.copy(rotateEnd);
          scope.update();
          break;
        case DOLLY:
          if (scope.enableZoom === false) return;
          dollyEnd.set(event.clientX, event.clientY);
          dollyDelta.subVectors(dollyEnd, dollyStart);
          if (dollyDelta.y > 0) dollyIn(getZoomScale(dollyDelta.y));
          else if (dollyDelta.y < 0) dollyOut(getZoomScale(dollyDelta.y));
          dollyStart.copy(dollyEnd);
          scope.update();
          break;
        case PAN:
          if (scope.enablePan === false) return;
          panEnd.set(event.clientX, event.clientY);
          panDelta.subVectors(panEnd, panStart);
          pan(panDelta.x, panDelta.y);
          panStart.copy(panEnd);
          scope.update();
          break;
      }
    }
    function onMouseWheel(event) {
      if (scope.enabled === false || scope.enableZoom === false || state !== -1) return;
      event.preventDefault();
      scope.dispatchEvent(_startEvent);
      if (event.deltaY < 0) dollyIn(getZoomScale(event.deltaY));
      else if (event.deltaY > 0) dollyOut(getZoomScale(event.deltaY));
      scope.update();
      scope.dispatchEvent(_endEvent);
    }
    function onKeyDown(event) {
      if (scope.enabled === false || scope.enablePan === false) return;
      switch (event.code) {
        case 'ArrowUp': pan(0, scope.keyPanSpeed); scope.update(); break;
        case 'ArrowDown': pan(0, -scope.keyPanSpeed); scope.update(); break;
        case 'ArrowLeft': pan(scope.keyPanSpeed, 0); scope.update(); break;
        case 'ArrowRight': pan(-scope.keyPanSpeed, 0); scope.update(); break;
      }
    }

    function onTouchStart(event) {
      trackPointer(event);
      switch (pointers.length) {
        case 1:
          rotateStart.set(event.pageX, event.pageY);
          state = ROTATE;
          break;
        case 2:
          var pos = getSecondPointerPosition(event);
          var x = 0.5 * (event.pageX + pos.x);
          var y = 0.5 * (event.pageY + pos.y);
          dollyStart.set(x, y);
          state = DOLLY;
          break;
      }
      scope.dispatchEvent(_startEvent);
    }
    function onTouchMove(event) {
      trackPointer(event);
      switch (state) {
        case ROTATE:
          rotateEnd.set(event.pageX, event.pageY);
          rotateDelta.subVectors(rotateEnd, rotateStart);
          var element = scope.domElement;
          rotateLeft(2 * Math.PI * rotateDelta.x / element.clientHeight * scope.rotateSpeed);
          rotateUp(2 * Math.PI * rotateDelta.y / element.clientHeight * scope.rotateSpeed);
          rotateStart.copy(rotateEnd);
          scope.update();
          break;
        case DOLLY:
          var pos = getSecondPointerPosition(event);
          var x = 0.5 * (event.pageX + pos.x);
          var y = 0.5 * (event.pageY + pos.y);
          dollyEnd.set(x, y);
          dollyDelta.subVectors(dollyEnd, dollyStart);
          if (dollyDelta.y > 0) dollyIn(getZoomScale(dollyDelta.y));
          else if (dollyDelta.y < 0) dollyOut(getZoomScale(dollyDelta.y));
          dollyStart.copy(dollyEnd);
          scope.update();
          break;
      }
    }
    function onContextMenu(event) {
      if (scope.enabled === false) return;
      event.preventDefault();
    }
    function addPointer(event) { pointers.push(event.pointerId); }
    function removePointer(event) {
      delete pointerPositions[event.pointerId];
      for (var i = 0; i < pointers.length; i++) {
        if (pointers[i] === event.pointerId) {
          pointers.splice(i, 1);
          return;
        }
      }
    }
    function trackPointer(event) {
      if (pointerPositions[event.pointerId] === undefined) {
        pointerPositions[event.pointerId] = new THREE.Vector2();
      }
      pointerPositions[event.pointerId].set(event.pageX, event.pageY);
    }
    function getSecondPointerPosition(event) {
      var pointerId = (event.pointerId === pointers[0]) ? pointers[1] : pointers[0];
      return pointerPositions[pointerId];
    }

    scope.domElement.addEventListener('contextmenu', onContextMenu, false);
    scope.domElement.addEventListener('pointerdown', onPointerDown, false);
    scope.domElement.addEventListener('wheel', onMouseWheel, { passive: false });
    scope.dispatchEvent({ type: 'change' });
  };

  OrbitControls.prototype = Object.create(THREE.EventDispatcher.prototype);
  OrbitControls.prototype.constructor = OrbitControls;

  root.THREE.OrbitControls = OrbitControls;

})(window);
