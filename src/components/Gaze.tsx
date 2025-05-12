import { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
	CirclePlayIcon,
	Loader2,
	NotepadText,
	StepForwardIcon,
} from "lucide-react";
import {
	GazeData,
	ParticipantStudyData,
	WebPageInfo,
	Point,
	VideoMetadata,
	WebcamMetadata,
	CurrentDisplayedPoint,
	UsedGroundTruthPoints,
	CheckedState,
} from "@/types/study";
import PhaseInstructions from "./PhaseInstructions";
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

type StudyCanvasProps = {
	setCurrentPhase: React.Dispatch<React.SetStateAction<number>>;
	setRecordedChunks: React.Dispatch<React.SetStateAction<Blob[]>>;
	participantStudyData: ParticipantStudyData;
	setParticipantStudyData: React.Dispatch<
		React.SetStateAction<ParticipantStudyData>
	>;
	dataCollectionDuration: number;
	pointAccelerationInitiationTime: number;
	baseTravelingPointSpeed: number;
	acceleratedTravelingPointSpeed: number;
	fixationDuration: number;
	travelDistance: number;
	requiredPointClickTimeLimit: number;
	setStartSimulation: React.Dispatch<React.SetStateAction<boolean>>;
};

function linearInterpolation(x: number, point1: Point, point2: Point) {
	return (
		point1.y + (x - point1.x) * ((point2.y - point1.y) / (point2.x - point1.x))
	);
}

function sleep(ms: number) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

export default function Gaze({
	setCurrentPhase,
	setRecordedChunks,
	participantStudyData,
	setParticipantStudyData,
	dataCollectionDuration,
	pointAccelerationInitiationTime,
	fixationDuration,
	baseTravelingPointSpeed,
	acceleratedTravelingPointSpeed,
	travelDistance,
	setStartSimulation,
	requiredPointClickTimeLimit,
}: StudyCanvasProps) {
	const [isInitializing, setIsInitializing] = useState<boolean>(true);
	const interactiveCanvasRef = useRef<HTMLCanvasElement>(null);

	// Webcam and face detector references
	const webcamRef = useRef<HTMLVideoElement>(null);
	const mediaRecorderRef = useRef<MediaRecorder | null>(null);
	const faceDetectorRef = useRef<FaceDetector | null>(null);
	const liveViewRef = useRef<HTMLDivElement>(null);
	const childrenRef = useRef<HTMLElement[]>([]);
	const lastVideoTime = useRef<number>(-1);
	const requestAnimationId = useRef<number | null>(null);

	const [showInstructions, setShowInstructions] = useState<boolean>(true);
	const [hasReadInstructions, setHasReadInstructions] =
		useState<CheckedState>(false);
	const [startDataCollection, setStartDataCollection] =
		useState<boolean>(false);
	const [gazeStudyCompleted, setGazeStudyCompleted] = useState<boolean>(false);

	const pointDistance = useRef<number>(50);
	const displayPointRadius = useRef<number>(14);

	const startDataCollectionRef = useRef<boolean>(false);
	const startingTimeStampMS = useRef<number>(0);
	const endingTimeStampMS = useRef<number>(0);
	const groundTruthRef = useRef<Point[]>([]);
	const collectedGazeDataRef = useRef<GazeData[]>([]);
	const unusedPointsIndexRef = useRef<number[]>([]);
	const currentPointRef = useRef<CurrentDisplayedPoint>({
		x: 0,
		y: 0,
		circle: new Path2D(),
		event: "move",
	});
	const usedGroundTruthPointsRef = useRef<UsedGroundTruthPoints[]>([]);
	const isPointAcceleratedRef = useRef<boolean>(false);

	const prevPointRef = useRef<Point | null>({
		x: Math.floor(window.innerWidth / 2),
		y: Math.floor(window.innerHeight / 2),
	});
	const nextPointRef = useRef<Point | null>(null);

	const webPageInfoRef = useRef<WebPageInfo>({
		width: window.innerWidth,
		height: window.innerHeight,
	});
	const videoMetaDataRef = useRef<VideoMetadata>({
		recording_start_timestamp: 0,
		recording_end_timestamp: 0,
	});
	const webcamMetadataRef = useRef<WebcamMetadata>({
		label: "",
		frame_rate: 0,
		resolution_width: 0,
		resolution_height: 0,
	});

	function resetStudy(reason: string) {
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state === "recording"
		) {
			mediaRecorderRef.current.stop();
			console.log("Recording stopped.");
		}

		if (webcamRef.current && webcamRef.current.srcObject) {
			const stream = webcamRef.current.srcObject as MediaStream;
			const tracks = stream.getTracks();
			tracks.forEach((track) => track.stop());
			webcamRef.current.srcObject = null;
		}

		toast.error(reason, {
			duration: 5000,
		});

		setRecordedChunks([]);
		setCurrentPhase(2);
	}

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

			if (detections.length === 0 && startDataCollectionRef.current) {
				resetStudy(
					"No face has been detected in the webcam. The study will restart."
				);

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

	const startRecording = useCallback(() => {
		if (webcamRef.current && webcamRef.current.srcObject) {
			const stream = webcamRef.current.srcObject as MediaStream;
			const mediaRecorder = new MediaRecorder(stream);
			mediaRecorderRef.current = mediaRecorder;

			if (
				!mediaRecorderRef.current ||
				mediaRecorderRef.current.state === "recording"
			)
				return;

			mediaRecorderRef.current.ondataavailable = (event) => {
				if (event.data.size > 0) {
					setRecordedChunks((prev) => [...prev, event.data]);
				}
			};

			mediaRecorderRef.current.start();

			const startTimestamp = performance.now();
			startingTimeStampMS.current = startTimestamp;
			videoMetaDataRef.current.recording_start_timestamp = startTimestamp;
			console.log(
				"Recording started at: ",
				startingTimeStampMS.current,
				startTimestamp
			);
		}
	}, [setRecordedChunks]);

	const stopRecording = useCallback(() => {
		if (
			mediaRecorderRef.current &&
			mediaRecorderRef.current.state === "recording"
		) {
			const stop = performance.now();
			const stopTimestamp = stop - startingTimeStampMS.current;
			videoMetaDataRef.current.recording_end_timestamp =
				stop - videoMetaDataRef.current.recording_start_timestamp;
			endingTimeStampMS.current = stopTimestamp;

			mediaRecorderRef.current.stop();
			console.log("Recording stopped at: ", stopTimestamp, stop);
		}
	}, []);

	const initializeGazePointsForCanvas = useCallback(() => {
		const canvas = interactiveCanvasRef.current;

		if (!canvas) {
			return;
		}

		const ctx = canvas.getContext("2d");

		if (!ctx) return;

		const padding = 30;
		const canvasWidth = ctx.canvas.width;
		const canvasHeight = ctx.canvas.height;

		webPageInfoRef.current = {
			width: canvasWidth,
			height: canvasHeight,
		};

		let currentX = padding;
		let currentY = padding;

		const points: Point[] = [{ x: currentX, y: currentY }];
		const pointIndices: number[] = [0];
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		while (currentX <= canvasWidth - padding && currentY <= canvasHeight) {
			if (
				currentX + pointDistance.current + displayPointRadius.current >=
				canvasWidth - padding
			) {
				currentX = padding;
				currentY += pointDistance.current + displayPointRadius.current;
			} else {
				currentX += pointDistance.current + displayPointRadius.current;
			}

			points.push({ x: currentX, y: currentY });
			pointIndices.push(points.length - 1);
		}

		groundTruthRef.current = points;
		unusedPointsIndexRef.current = pointIndices;

		// Draw the initial point in the center of the canvas
		const centerX = Math.floor(canvasWidth / 2);
		const centerY = Math.floor(canvasHeight / 2);
		const circle = new Path2D();
		circle.arc(centerX, centerY, displayPointRadius.current, 0, 2 * Math.PI);
		ctx.fillStyle = "green";
		ctx.fill(circle);
	}, []);

	const synchronizePointFixationDuration = async (
		x: number,
		y: number,
		ctx: CanvasRenderingContext2D,
		duration: number,
		speed: string
	) => {
		/*
			Point gaze fixation duration has 2 required mouse clicks at the beginning and end duration interval:
		*/

		collectedGazeDataRef.current.push({
			x,
			y,
			timestamp: performance.now() - startingTimeStampMS.current,
			event: "start_fixation",
			speed: speed,
		});

		const requiredClickDuration = requiredPointClickTimeLimit;
		const intervalMiddleDuration = duration;

		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
		const circle = new Path2D();
		circle.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
		ctx.fillStyle = "red";
		ctx.fill(circle);

		const startPointFixationClickTimestamp =
			performance.now() - startingTimeStampMS.current;

		if (currentPointRef.current) {
			currentPointRef.current.event = "require_starting_mouse_click";
		}

		collectedGazeDataRef.current.push({
			x,
			y,
			timestamp: startPointFixationClickTimestamp,
			event: "require_starting_mouse_click",
			speed: speed,
		});

		await new Promise<void>((resolve) => {
			setTimeout(() => {
				resolve();
			}, requiredClickDuration);
		});

		if (usedGroundTruthPointsRef.current.length === 0) {
			return;
		}

		if (
			!usedGroundTruthPointsRef.current[
				usedGroundTruthPointsRef.current.length - 1
			].interval_start_click
		) {
			resetStudy(
				"You did not click on all the points. The study will restart."
			);
			return;
		}

		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
		const fixationPeriodCircle = new Path2D();
		fixationPeriodCircle.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
		ctx.fillStyle = "green";
		ctx.fill(fixationPeriodCircle);

		await new Promise<void>((resolve) => {
			setTimeout(() => {
				resolve();
			}, intervalMiddleDuration);
		});

		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
		const endingFixationClickCircle = new Path2D();
		endingFixationClickCircle.arc(
			x,
			y,
			displayPointRadius.current,
			0,
			2 * Math.PI
		);
		ctx.fillStyle = "red";
		ctx.fill(endingFixationClickCircle);

		const endFixationClickTimestamp =
			performance.now() - startingTimeStampMS.current;

		if (currentPointRef.current) {
			currentPointRef.current.event = "require_ending_mouse_click";
		}

		collectedGazeDataRef.current.push({
			x,
			y,
			timestamp: endFixationClickTimestamp,
			event: "require_ending_mouse_click",
			speed: speed,
		});

		await new Promise<void>((resolve) => {
			setTimeout(() => {
				resolve();
			}, requiredClickDuration);
		});

		if (usedGroundTruthPointsRef.current.length === 0) {
			return;
		}

		if (
			!usedGroundTruthPointsRef.current[
				usedGroundTruthPointsRef.current.length - 1
			].interval_end_click
		) {
			resetStudy(
				"You did not click on all the points. The study will restart."
			);
			return;
		}

		collectedGazeDataRef.current.push({
			x,
			y,
			timestamp: performance.now() - startingTimeStampMS.current,
			event: "end_fixation",
			speed: speed,
		});

		gazeCollectionLoop();
	};

	// eslint-disable-next-line react-hooks/exhaustive-deps
	async function acceleratedPointDisplayOnCanvas() {
		const canvas = interactiveCanvasRef.current;

		if (!canvas) {
			return;
		}

		const ctx = canvas.getContext("2d");

		if (!ctx) {
			return;
		}

		if (!nextPointRef.current) return;

		const { x, y } = nextPointRef.current;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		const circle = new Path2D();
		ctx.fillStyle = "green";
		circle.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
		ctx.fill(circle);

		// const pointDisplayedTimestamp =
		// 	performance.now() - startingTimeStampMS.current;

		currentPointRef.current = {
			x,
			y,
			circle,
			event: "stationary",
		};

		usedGroundTruthPointsRef.current.push({
			x,
			y,
			speed: "accelerated",
			interval_start_click: false,
			interval_end_click: false,
			interval_start_click_timestamp: 0,
			interval_end_click_timestamp: 0,
		});

		// collectedGazeDataRef.current.push({
		// 	x,
		// 	y,
		// 	timestamp: pointDisplayedTimestamp,
		// 	event: "stationary",
		// 	speed: "accelerated",
		// });

		await synchronizePointFixationDuration(
			x,
			y,
			ctx,
			fixationDuration,
			"accelerated"
		);
	}

	// eslint-disable-next-line react-hooks/exhaustive-deps
	async function basePointDisplayOnCanvas() {
		const canvas = interactiveCanvasRef.current;

		if (!canvas) {
			return;
		}

		const ctx = canvas.getContext("2d");

		if (!ctx) {
			return;
		}

		if (!nextPointRef.current) return;

		const { x, y } = nextPointRef.current;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		const circle = new Path2D();
		ctx.fillStyle = "green";
		circle.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
		ctx.fill(circle);

		// const pointDisplayedTimestamp =
		// 	performance.now() - startingTimeStampMS.current;

		currentPointRef.current = {
			x,
			y,
			circle,
			event: "stationary",
		};

		usedGroundTruthPointsRef.current.push({
			x,
			y,
			speed: "normal",
			interval_start_click: false,
			interval_end_click: false,
			interval_start_click_timestamp: 0,
			interval_end_click_timestamp: 0,
		});

		// collectedGazeDataRef.current.push({
		// 	x,
		// 	y,
		// 	timestamp: pointDisplayedTimestamp,
		// 	event: "stationary",
		// 	speed: "normal",
		// });

		await synchronizePointFixationDuration(
			x,
			y,
			ctx,
			fixationDuration,
			"normal"
		);
	}

	const moveToNextPoint = useCallback(
		async (
			canvas: HTMLCanvasElement,
			ctx: CanvasRenderingContext2D,
			movingPointDuration: number,
			prevX: number,
			prevY: number,
			nextX: number,
			nextY: number,
			travelingDistance: number
		) => {
			if (prevX < nextX) {
				for (
					let xCoord = prevX + travelingDistance;
					xCoord < nextX;
					xCoord += travelingDistance
				) {
					const yCoord = Math.floor(
						linearInterpolation(
							xCoord,
							{ x: prevX, y: prevY },
							{ x: nextX, y: nextY }
						)
					);

					await drawMovingPoint(
						canvas,
						ctx,
						xCoord,
						yCoord,
						movingPointDuration
					);
				}
			} else if (prevX > nextX) {
				for (
					let xCoord = prevX - travelingDistance;
					xCoord > nextX;
					xCoord -= travelingDistance
				) {
					const yCoord = Math.floor(
						linearInterpolation(
							xCoord,
							{ x: prevX, y: prevY },
							{ x: nextX, y: nextY }
						)
					);

					await drawMovingPoint(
						canvas,
						ctx,
						xCoord,
						yCoord,
						movingPointDuration
					);
				}
			} else if (prevY < nextY) {
				for (
					let yCoord = prevY + travelingDistance;
					yCoord < nextY;
					yCoord += travelingDistance
				) {
					await drawMovingPoint(
						canvas,
						ctx,
						nextX,
						yCoord,
						movingPointDuration
					);
				}
			} else if (prevY > nextY) {
				for (
					let yCoord = prevY - travelingDistance;
					yCoord > nextY;
					yCoord -= travelingDistance
				) {
					await drawMovingPoint(
						canvas,
						ctx,
						nextX,
						yCoord,
						movingPointDuration
					);
				}
			}
		},
		[]
	);

	async function drawMovingPoint(
		canvas: HTMLCanvasElement,
		ctx: CanvasRenderingContext2D,
		x: number,
		y: number,
		movingPointDuration: number
	) {
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		const interpolatedPoint = new Path2D();
		interpolatedPoint.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
		ctx.fillStyle = "green";
		ctx.fill(interpolatedPoint);

		const pointDisplayedTimestamp =
			performance.now() - startingTimeStampMS.current;

		if (currentPointRef.current) {
			currentPointRef.current.event = "move";
		}

		collectedGazeDataRef.current.push({
			x,
			y,
			timestamp: pointDisplayedTimestamp,
			event: "move",
			speed: isPointAcceleratedRef.current ? "accelerated" : "normal",
		});

		await sleep(movingPointDuration);
	}

	const gazeCollectionLoop = useCallback(async () => {
		if (
			performance.now() - startingTimeStampMS.current >=
			dataCollectionDuration
		) {
			async function fullScreenScan() {
				const canvas = interactiveCanvasRef.current;

				if (!canvas) {
					return;
				}

				const ctx = canvas.getContext("2d");

				if (!ctx) {
					return;
				}

				const padding = 30;
				const canvasWidth = ctx.canvas.width;
				const canvasHeight = ctx.canvas.height;

				const topLeftPoint = { x: padding, y: padding };
				const topRightPoint = { x: canvasWidth - padding, y: padding };
				const bottomLeftPoint = { x: padding, y: canvasHeight - padding };
				const bottomRightPoint = {
					x: canvasWidth - padding,
					y: canvasHeight - padding,
				};
				const centerPoint = {
					x: Math.floor(canvasWidth / 2),
					y: Math.floor(canvasHeight / 2),
				};

				const pointOrder = [
					centerPoint,
					topLeftPoint,
					// bottomLeftPoint,
					// topRightPoint,
					// bottomRightPoint,
					// topLeftPoint,
					// topRightPoint,
					// bottomRightPoint,
					// bottomLeftPoint,
					// topLeftPoint,
				];

				if (!prevPointRef.current) {
					prevPointRef.current = centerPoint;
				}

				isPointAcceleratedRef.current = true;

				for (const point of pointOrder) {
					nextPointRef.current = point;

					await moveToNextPoint(
						canvas,
						ctx,
						acceleratedTravelingPointSpeed,
						prevPointRef.current.x,
						prevPointRef.current.y,
						point.x,
						point.y,
						20
					);

					prevPointRef.current = point;

					const { x, y } = nextPointRef.current;

					ctx.clearRect(0, 0, canvas.width, canvas.height);

					const currPoint = new Path2D();
					ctx.fillStyle = "green";
					currPoint.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
					ctx.fill(currPoint);

					const displayedTimestamp =
						performance.now() - startingTimeStampMS.current;

					currentPointRef.current = {
						x,
						y,
						circle: currPoint,
						event: "stationary",
					};

					usedGroundTruthPointsRef.current.push({
						x,
						y,
						speed: "accelerated",
						interval_start_click: false,
						interval_end_click: false,
						interval_start_click_timestamp: 0,
						interval_end_click_timestamp: 0,
					});

					collectedGazeDataRef.current.push({
						x,
						y,
						timestamp: displayedTimestamp,
						event: "start_fixation",
						speed: "accelerated",
					});

					const requiredClickDuration = requiredPointClickTimeLimit;
					const intervalMiddleDuration = fixationDuration;

					ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
					const circle = new Path2D();
					circle.arc(x, y, displayPointRadius.current, 0, 2 * Math.PI);
					ctx.fillStyle = "red";
					ctx.fill(circle);

					const startPointFixationClickTimestamp =
						performance.now() - startingTimeStampMS.current;

					if (currentPointRef.current) {
						currentPointRef.current.event = "require_starting_mouse_click";
					}

					collectedGazeDataRef.current.push({
						x,
						y,
						timestamp: startPointFixationClickTimestamp,
						event: "require_starting_mouse_click",
						speed: "accelerated",
					});

					await new Promise<void>((resolve) => {
						setTimeout(() => {
							resolve();
						}, requiredClickDuration);
					});

					if (usedGroundTruthPointsRef.current.length === 0) {
						return;
					}

					if (
						!usedGroundTruthPointsRef.current[
							usedGroundTruthPointsRef.current.length - 1
						].interval_start_click
					) {
						resetStudy(
							"You did not click on all the points. The study will restart."
						);
						return;
					}

					ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
					const fixationPeriodCircle = new Path2D();
					fixationPeriodCircle.arc(
						x,
						y,
						displayPointRadius.current,
						0,
						2 * Math.PI
					);
					ctx.fillStyle = "green";
					ctx.fill(fixationPeriodCircle);

					await new Promise<void>((resolve) => {
						setTimeout(() => {
							resolve();
						}, intervalMiddleDuration);
					});

					ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
					const endingFixationClickCircle = new Path2D();
					endingFixationClickCircle.arc(
						x,
						y,
						displayPointRadius.current,
						0,
						2 * Math.PI
					);
					ctx.fillStyle = "red";
					ctx.fill(endingFixationClickCircle);

					const endFixationClickTimestamp =
						performance.now() - startingTimeStampMS.current;

					if (currentPointRef.current) {
						currentPointRef.current.event = "require_ending_mouse_click";
					}

					collectedGazeDataRef.current.push({
						x,
						y,
						timestamp: endFixationClickTimestamp,
						event: "require_ending_mouse_click",
						speed: "accelerated",
					});

					await new Promise<void>((resolve) => {
						setTimeout(() => {
							resolve();
						}, requiredClickDuration);
					});

					if (usedGroundTruthPointsRef.current.length === 0) {
						return;
					}

					if (
						!usedGroundTruthPointsRef.current[
							usedGroundTruthPointsRef.current.length - 1
						].interval_end_click
					) {
						resetStudy(
							"You did not click on all the points. The study will restart."
						);
						return;
					}

					collectedGazeDataRef.current.push({
						x,
						y,
						timestamp: displayedTimestamp,
						event: "end_fixation",
						speed: "accelerated",
					});
				}
			}

			await fullScreenScan();

			stopRecording();
			if (webcamRef.current && webcamRef.current.srcObject) {
				const stream = webcamRef.current.srcObject as MediaStream;
				const tracks = stream.getTracks();
				tracks.forEach((track) => track.stop());
				webcamRef.current.srcObject = null;
			}

			if (!liveViewRef.current) return;

			for (const child of childrenRef.current) {
				liveViewRef.current.removeChild(child);
			}
			childrenRef.current.splice(0);

			if (!faceDetectorRef.current) return;

			faceDetectorRef.current.close();

			setGazeStudyCompleted(true);
			return;
		}

		const canvas = interactiveCanvasRef.current;

		if (!canvas) {
			return;
		}

		const ctx = canvas.getContext("2d");

		if (!ctx) {
			return;
		}

		if (unusedPointsIndexRef.current.length === 0) {
			unusedPointsIndexRef.current = groundTruthRef.current.map((_, i) => i);
		}

		const randomIndex = Math.floor(
			Math.random() * (unusedPointsIndexRef.current.length - 1)
		);

		const selectedPointIndex = unusedPointsIndexRef.current[randomIndex];

		const { x: nextX, y: nextY } = groundTruthRef.current[selectedPointIndex];

		nextPointRef.current = { x: nextX, y: nextY };

		if (!prevPointRef.current) return;

		const { x: prevX, y: prevY } = prevPointRef.current;

		if (
			performance.now() - startingTimeStampMS.current >=
				pointAccelerationInitiationTime &&
			!isPointAcceleratedRef.current
		) {
			isPointAcceleratedRef.current = true;
		}

		const movingPointDuration = isPointAcceleratedRef.current
			? acceleratedTravelingPointSpeed
			: baseTravelingPointSpeed;
		// const movingPointDuration = isPointAcceleratedRef.current ? 500 : 800;

		const travelingDistance = travelDistance + displayPointRadius.current;
		// const travelingDistance = pointDistance.current + displayPointRadius.current;

		await moveToNextPoint(
			canvas,
			ctx,
			movingPointDuration,
			prevX,
			prevY,
			nextX,
			nextY,
			travelingDistance
		);

		prevPointRef.current = { x: nextX, y: nextY };

		const filteredPoints = unusedPointsIndexRef.current.filter(
			(value) => value !== unusedPointsIndexRef.current[randomIndex]
		);

		unusedPointsIndexRef.current = filteredPoints;

		if (isPointAcceleratedRef.current) {
			acceleratedPointDisplayOnCanvas();
		} else {
			basePointDisplayOnCanvas();
		}
	}, [
		pointAccelerationInitiationTime,
		acceleratedPointDisplayOnCanvas,
		basePointDisplayOnCanvas,
		dataCollectionDuration,
		stopRecording,
		travelDistance,
		baseTravelingPointSpeed,
		acceleratedTravelingPointSpeed,
		moveToNextPoint,
		fixationDuration,
	]);

	useEffect(() => {
		async function getWebcamFeed() {
			try {
				faceDetectorRef.current = await initializefaceDetector();
				if (!faceDetectorRef.current) {
					toast.error(
						"An error has occured please contact the study administrator.",
						{
							duration: 6000,
						}
					);

					startDataCollectionRef.current = false;
					setStartSimulation(false);
					setCurrentPhase(0);
					return;
				}
			} catch (error) {
				toast.error(
					"An error has occured please contact the study administrator.",
					{
						duration: 6000,
					}
				);
				setStartSimulation(false);
				setCurrentPhase(0);
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

				if (stream.getVideoTracks().length > 0) {
					const videoTrack = stream.getVideoTracks()[0];
					const settings = videoTrack.getSettings();

					webcamMetadataRef.current = {
						label: videoTrack.label,
						frame_rate: settings.frameRate ? settings.frameRate : 0,
						resolution_width: settings.width ? settings.width : 0,
						resolution_height: settings.height ? settings.height : 0,
					};
				}

				setIsInitializing(false);
			} catch (err) {
				toast.error(
					"Unable to access webcam, please make sure that you have a webcam connected and that you have given permission to access it.",
					{
						duration: 6000,
					}
				);
				console.error("Error accessing webcam: ", err);

				startDataCollectionRef.current = false;
				setStartSimulation(false);
				setCurrentPhase(0);
				return;
			}
		}

		getWebcamFeed();

		return () => {
			if (requestAnimationId.current) {
				window.cancelAnimationFrame(requestAnimationId.current);
			}

			if (!liveViewRef.current) return;

			for (const child of childrenRef.current) {
				liveViewRef.current.removeChild(child);
			}
			childrenRef.current.splice(0);

			if (!faceDetectorRef.current) return;

			faceDetectorRef.current.close();

			if (webcamRef.current && webcamRef.current.srcObject) {
				const stream = webcamRef.current.srcObject as MediaStream;
				const tracks = stream.getTracks();
				tracks.forEach((track) => track.stop());
				webcamRef.current.srcObject = null;
			}
		};
	}, [setStartSimulation]);

	useEffect(() => {
		async function run() {
			initializeGazePointsForCanvas();
			toast.info(
				"The study has started get ready to for the point to move and click when it turns red.",
				{ duration: 4000 }
			);
			await sleep(6000);

			if (!startDataCollectionRef.current) {
				return;
			}

			startRecording();

			if (!startDataCollectionRef.current) {
				return;
			}

			gazeCollectionLoop();
		}

		if (startDataCollection && !gazeStudyCompleted) {
			run();
		}
	}, [
		startDataCollection,
		gazeStudyCompleted,
		gazeCollectionLoop,
		startRecording,
		initializeGazePointsForCanvas,
	]);

	function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
		if (!startDataCollectionRef.current) {
			return;
		}

		const xClick = e.clientX;
		const yClick = e.clientY;

		const canvas = interactiveCanvasRef.current;

		if (!canvas) {
			return;
		}

		const ctx = canvas.getContext("2d");

		if (!ctx || !currentPointRef.current || !currentPointRef.current.circle) {
			return;
		}

		if (usedGroundTruthPointsRef.current.length === 0) {
			return;
		}

		if (
			currentPointRef.current.event !== "require_starting_mouse_click" &&
			currentPointRef.current.event !== "require_ending_mouse_click"
		) {
			return;
		}

		if (
			currentPointRef.current.event === "require_starting_mouse_click" &&
			usedGroundTruthPointsRef.current[
				usedGroundTruthPointsRef.current.length - 1
			].interval_start_click
		) {
			return;
		}

		if (
			currentPointRef.current.event === "require_ending_mouse_click" &&
			usedGroundTruthPointsRef.current[
				usedGroundTruthPointsRef.current.length - 1
			].interval_end_click
		) {
			return;
		}

		if (ctx.isPointInPath(currentPointRef.current.circle, xClick, yClick)) {
			const clickTimestamp = performance.now() - startingTimeStampMS.current;

			collectedGazeDataRef.current.push({
				x: currentPointRef.current.x,
				y: currentPointRef.current.y,
				timestamp: clickTimestamp,
				event: "mouse_click",
				speed: isPointAcceleratedRef.current ? "accelerated" : "normal",
			});

			if (currentPointRef.current.event === "require_starting_mouse_click") {
				usedGroundTruthPointsRef.current[
					usedGroundTruthPointsRef.current.length - 1
				].interval_start_click = true;

				usedGroundTruthPointsRef.current[
					usedGroundTruthPointsRef.current.length - 1
				].interval_start_click_timestamp = clickTimestamp;
			} else if (
				currentPointRef.current.event === "require_ending_mouse_click"
			) {
				usedGroundTruthPointsRef.current[
					usedGroundTruthPointsRef.current.length - 1
				].interval_end_click = true;

				usedGroundTruthPointsRef.current[
					usedGroundTruthPointsRef.current.length - 1
				].interval_end_click_timestamp = clickTimestamp;
			}

			ctx.clearRect(0, 0, canvas.width, canvas.height);

			const circle = new Path2D();
			circle.arc(
				currentPointRef.current.x,
				currentPointRef.current.y,
				displayPointRadius.current,
				0,
				2 * Math.PI
			);
			ctx.fillStyle = "green";
			ctx.fill(circle);
		}
	}

	if (gazeStudyCompleted) {
		return (
			<div className="w-full h-screen flex flex-col items-center justify-center">
				<p className="text-2xl font-medium">
					This portion of the study has been completed.
				</p>
				<Button
					onClick={() => {
						// const data = {
						// 	gaze_study_data: {
						// 		study_duration: dataCollectionDuration,
						// 		screen_width: screen.width,
						// 		screen_height: screen.height,
						// 		window_inner_height: window.innerHeight,
						// 		window_inner_width: window.innerWidth,
						// 		window_outer_height: window.outerHeight,
						// 		window_outer_width: window.outerWidth,
						// 		gaze_data: collectedGazeDataRef.current,
						// 		usedGroundTruthPointsRef: usedGroundTruthPointsRef.current,
						// 		// point_logs: pointLogsRef.current,
						// 		study_start_timestamp: 0,
						// 		study_end_timestamp: endingTimeStampMS.current,
						// 		webcam_metadata: webcamMetadataRef.current,
						// 	},
						// 	video_recording_metadata: {
						// 		file_name: `${participantStudyData.participant_id}.webm`,
						// 		recording_start_timestamp: 0,
						// 		recording_end_timestamp:
						// 			videoMetaDataRef.current.recording_end_timestamp,
						// 	},
						// };

						// console.log(data);

						setParticipantStudyData((prev) => ({
							...prev,
							gaze_study_data: {
								study_duration: endingTimeStampMS.current,
								screen_width: screen.width,
								screen_height: screen.height,
								window_inner_height: window.innerHeight,
								window_inner_width: window.innerWidth,
								window_outer_height: window.outerHeight,
								window_outer_width: window.outerWidth,
								gaze_data: collectedGazeDataRef.current,
								used_ground_truth_points: usedGroundTruthPointsRef.current,
								// point_logs: pointLogsRef.current,
								study_start_timestamp: 0,
								study_end_timestamp: endingTimeStampMS.current,
								webcam_metadata: webcamMetadataRef.current,
							},
							video_recording_metadata: {
								file_name: `${participantStudyData.participant_id}.webm`,
								recording_start_timestamp: 0,
								recording_end_timestamp:
									videoMetaDataRef.current.recording_end_timestamp,
							},
						}));

						setCurrentPhase((prev) => prev + 1);
					}}
					className="mt-4 flex items-center gap-2"
				>
					<StepForwardIcon /> Continue to next phase of the study
				</Button>
			</div>
		);
	}

	return (
		<>
			<div
				className={`absolute top-0 left-0 h-80 w-80 
					${startDataCollection ? "opacity-0" : "opacity-100"}
				`}
			>
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

			{isInitializing && (
				<div className="w-full h-screen flex items-center justify-center">
					<Loader2 className="mr-2 animate-spin h-7 w-7" />
					<p className="text-2xl font-medium">Initializing...</p>
				</div>
			)}

			{startDataCollection && !isInitializing ? (
				<canvas
					ref={interactiveCanvasRef}
					height={window.innerHeight}
					width={window.innerWidth}
					className="m-0 p-0 z-10"
					onClick={handleCanvasClick}
				/>
			) : (
				!isInitializing && (
					<>
						<PhaseInstructions
							title="Study Instructions"
							open={showInstructions}
							setOpen={setShowInstructions}
							hasReadInstructions={hasReadInstructions}
							setHasReadInstructions={setHasReadInstructions}
						>
							<div className="text-gray-800">
								<p className="mb-2">
									This is the actual portion of the study where the data
									collection will take place and will take approximately{" "}
									<strong>5 to 7 minutes</strong>. Your face and eye movement
									will be recorded during this study. Before beginning, please
									ensure the following:
								</p>

								<ul className="list-disc pl-5 mb-4">
									<li>
										Your <strong>webcam video feed</strong> is visible in the
										top left corner of the screen as it will be recorded.
									</li>
									<li>
										Your <strong>face is clearly visible</strong>, centered in
										the video feed, and you are in a well-lit area.
									</li>
									<li>
										<strong>Maximize the browser window</strong> and do not
										resize it or switch to another tab during the study.
									</li>
									<li>
										Please stay seated for the entire duration of the study and
										avoid any distractions or display excessive movement.
									</li>
								</ul>

								<p className="mb-2">
									<strong>How the Study Works:</strong>
								</p>

								<ul className="list-disc pl-5 mb-4">
									<li>
										Please focus your eyes on the moving point on the screen for
										the entire duration of the study.
									</li>
									<li>
										When the study starts a{" "}
										<span className="text-green-600 font-bold">green dot</span>{" "}
										will appear in the <strong>center</strong> of the screen.
										This dot will continuously move to a new location on the
										screen, always focus your attention on this dot.
									</li>
									<li>
										When the dot stops moving and turns{" "}
										<span className="text-red-600 font-bold">red</span>,{" "}
										<strong>click on it</strong>. Successfully clicking on the
										dot will turn it{" "}
										<span className="text-green-600 font-bold">green</span>{" "}
										again. You will have to do this <strong>twice</strong> for
										each point.
									</li>
									<li>When 1 minute remains the point will move faster.</li>
								</ul>

								<p className="mb-2 text-lg font-bold underline">Important:</p>

								<ul className="list-disc pl-5 mb-4 font-bold ">
									<li className="text-red-600">
										If you fail to click all the points when they turn red, the
										study will restart from the beginning.
									</li>
									<li className="text-red-600">
										If your face is not visible and detected in the webcam feed,
										the study will restart from the beginning.
									</li>
									<li className="text-red-600">
										Do not resize the window or switch browser tabs, as this
										will disrupt the study and the study will restart from the
										beginning.
									</li>
								</ul>
							</div>
						</PhaseInstructions>

						<div className="flex flex-col items-center justify-center h-screen w-full p-6 bg-white rounded-lg">
							<div className="max-w-3xl">
								<h1 className="text-3xl font-semibold mb-6 text-gray-900 text-center">
									Please ensure your face is centered in the webcam at all times
									and you are in a brightly lit area.
								</h1>
							</div>

							<Button
								onClick={() => {
									setHasReadInstructions(false);
									setShowInstructions(true);
								}}
								className="flex items-center text-base px-4 py-5 mb-4"
							>
								<NotepadText className="h-6 w-6 mr-2" />
								Instructions
							</Button>

							<Button
								onClick={() => {
									toast.dismiss();
									startDataCollectionRef.current = true;
									setStartDataCollection(true);
								}}
								className="flex items-center text-base px-4 py-5"
							>
								<CirclePlayIcon className="h-6 w-6 mr-2" />
								Start Study
							</Button>
						</div>
					</>
				)
			)}
		</>
	);
}
