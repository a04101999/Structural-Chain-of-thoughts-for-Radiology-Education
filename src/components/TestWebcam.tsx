import { useEffect, useRef, useCallback } from "react";
import { toast } from "sonner";
import {
	FaceDetector,
	FilesetResolver,
	Detection,
} from "@mediapipe/tasks-vision";

const initializefaceDetector = async () => {
	const vision = await FilesetResolver.forVisionTasks(
		"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
	);

	return await FaceDetector.createFromOptions(vision, {
		baseOptions: {
			modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
			delegate: "CPU",
		},
		runningMode: "VIDEO",
	});
};

type TestWebcamProps = {
	isTestingWebcam: boolean;
	setIsTestingWebcam: React.Dispatch<React.SetStateAction<boolean>>;
};

export default function TestWebcam({
	isTestingWebcam,
	setIsTestingWebcam,
}: TestWebcamProps) {
	const webcamRef = useRef<HTMLVideoElement>(null);
	const faceDetectorRef = useRef<FaceDetector | null>(null);
	const liveViewRef = useRef<HTMLDivElement>(null);
	const childrenRef = useRef<HTMLElement[]>([]);
	const lastVideoTime = useRef<number>(-1);
	const faceUndetectedToastId = useRef<string | number | undefined>(undefined);
	const requestAnimationId = useRef<number | null>(null);

	const predictWebcam = useCallback(async () => {
		// if image mode is initialized, create a new classifier with video runningMode
		if (!webcamRef.current) return;
		if (!faceDetectorRef.current) return;

		function displayVideoDetections(detections: Detection[]) {
			if (!liveViewRef.current) return;
			if (!webcamRef.current) return;

			// Remove any highlighting from previous frame.
			for (const child of childrenRef.current) {
				liveViewRef.current.removeChild(child);
			}
			childrenRef.current.splice(0);

			if (detections.length === 0) {
				if (!faceUndetectedToastId.current) {
					faceUndetectedToastId.current = "face-undetected";

					toast.warning("No face detected", {
						onAutoClose: () => {
							faceUndetectedToastId.current = undefined;
							// console.log("Toast auto closed");
							// console.log(faceUndetectedToastId.current);
						},
						onDismiss: () => {
							faceUndetectedToastId.current = undefined;
							// console.log("Toast auto closed");
							// console.log(faceUndetectedToastId.current);
						},
						duration: 8000,
					});
				}

				return;
			}

			const detection = detections[0];
			if (!detection || !detection.boundingBox) return;

			// Store drawn objects in memory so they are queued to delete at next call
			const leftBoundFace = {
				x: -1,
				y: -1,
			};
			const rightBoundFace = {
				x: -1,
				y: -1,
			};

			for (const keypoint of detection.keypoints) {
				const keypointEl = document.createElement("span");
				keypointEl.className = "key-point";
				keypointEl.style.top = `${
					keypoint.y * webcamRef.current.offsetHeight - 3
				}px`;
				keypointEl.style.left = `${
					webcamRef.current.offsetWidth -
					keypoint.x * webcamRef.current.offsetWidth -
					3
				}px`;

				liveViewRef.current.appendChild(keypointEl);
				childrenRef.current.push(keypointEl);

				const xCoord =
					webcamRef.current.offsetWidth -
					keypoint.x * webcamRef.current.offsetWidth -
					3;
				const yCoord = keypoint.y * webcamRef.current.offsetHeight - 3;

				if (leftBoundFace.x === -1) {
					(leftBoundFace.x =
						webcamRef.current.offsetWidth -
						keypoint.x * webcamRef.current.offsetWidth -
						3),
						(leftBoundFace.y = keypoint.y * webcamRef.current.offsetHeight - 3);

					(rightBoundFace.x =
						webcamRef.current.offsetWidth -
						keypoint.x * webcamRef.current.offsetWidth -
						3),
						(rightBoundFace.y =
							keypoint.y * webcamRef.current.offsetHeight - 3);
				} else if (xCoord < leftBoundFace.x) {
					leftBoundFace.x = xCoord;
					leftBoundFace.y = yCoord;
				} else if (xCoord > rightBoundFace.x) {
					rightBoundFace.x = xCoord;
					rightBoundFace.y = yCoord;
				}
			}

			const highlighter = document.createElement("div");
			highlighter.setAttribute("class", "highlighter");
			highlighter.style.left = `${leftBoundFace.x - 10}px`;
			highlighter.style.top = `${leftBoundFace.y - 80}px`;
			highlighter.style.width = `${rightBoundFace.x - leftBoundFace.x + 30}px`;
			highlighter.style.height = `${
				detection.boundingBox.height *
					(webcamRef.current.height / webcamRef.current.videoHeight) +
				30
			}px`;
			liveViewRef.current.appendChild(highlighter);
			childrenRef.current.push(highlighter);
		}

		const startTimeMs = performance.now();

		// Detect faces using detectForVideo
		if (webcamRef.current.currentTime !== lastVideoTime.current) {
			lastVideoTime.current = webcamRef.current.currentTime;
			const detections = faceDetectorRef.current.detectForVideo(
				webcamRef.current,
				startTimeMs
			).detections;
			displayVideoDetections(detections);
		}

		// Call this function again to keep predicting when the browser is ready
		requestAnimationId.current = window.requestAnimationFrame(predictWebcam);
	}, []);

	const getWebcamFeed = useCallback(async () => {
		try {
			faceDetectorRef.current = await initializefaceDetector();
			if (!faceDetectorRef.current) {
				toast.error(
					"An error has occured please contact the study administrator.",
					{
						duration: 6000,
					}
				);
				setIsTestingWebcam(false);
				return;
			}
		} catch (error) {
			toast.error(
				"An error has occured please contact the study administrator.",
				{
					duration: 6000,
				}
			);
			setIsTestingWebcam(false);
			return;
		}

		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				video: true,
				audio: false,
			});

			if (webcamRef.current) {
				webcamRef.current.srcObject = stream;
				webcamRef.current.addEventListener("loadeddata", predictWebcam);
			}
		} catch (err) {
			toast.error(
				"Unable to access webcam, please make sure that you have a webcam connected and that you have given permission to access it.",
				{
					duration: 6000,
				}
			);
			console.error("Error accessing webcam: ", err);
			setIsTestingWebcam(false);
		}
	}, [predictWebcam]);

	const stopWebcamFeed = useCallback(() => {
		if (webcamRef.current && webcamRef.current.srcObject) {
			const stream = webcamRef.current.srcObject as MediaStream;
			const tracks = stream.getTracks();
			tracks.forEach((track) => track.stop());
			webcamRef.current.srcObject = null;
		}
	}, []);

	useEffect(() => {
		if (isTestingWebcam) {
			getWebcamFeed();
		} else {
			stopWebcamFeed();
		}

		return () => {
			if (requestAnimationId.current) {
				window.cancelAnimationFrame(requestAnimationId.current);
			}

			stopWebcamFeed();

			if (!liveViewRef.current) return;

			for (const child of childrenRef.current) {
				liveViewRef.current.removeChild(child);
			}
			childrenRef.current.splice(0);

			if (!faceDetectorRef.current) return;

			faceDetectorRef.current.close();
		};
	}, [isTestingWebcam, getWebcamFeed, stopWebcamFeed]);

	return (
		<div className="absolute top-0 left-0 h-80 w-80">
			<div ref={liveViewRef} id="liveView">
				<video
					id="webcam"
					ref={webcamRef}
					autoPlay
					muted
					playsInline
					width={320}
					height={320}
				/>
			</div>
		</div>
	);
}
