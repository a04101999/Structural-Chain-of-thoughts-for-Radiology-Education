import { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { CirclePlayIcon, Loader2, NotepadText } from "lucide-react";
import {
	GazeData,
	ParticipantStudyData,
	WebPageInfo,
	Point,
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
	participantStudyData: ParticipantStudyData;
	dataCollectionDuration: number;
	pointAccelerationInitiationTime: number;
	baseTravelingPointSpeed: number;
	acceleratedTravelingPointSpeed: number;
	fixationDuration: number;
	travelDistance: number;
	requiredPointClickTimeLimit: number;
	setStartDemo: React.Dispatch<React.SetStateAction<boolean>>;
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

export default function Demo({
	setCurrentPhase,
	dataCollectionDuration,
	pointAccelerationInitiationTime,
	fixationDuration,
	baseTravelingPointSpeed,
	acceleratedTravelingPointSpeed,
	travelDistance,
	setStartDemo,
	requiredPointClickTimeLimit,
}: StudyCanvasProps) {
	const [isInitializing, setIsInitializing] = useState<boolean>(true);
	const interactiveCanvasRef = useRef<HTMLCanvasElement>(null);

	// Webcam and face detector references
	const webcamRef = useRef<HTMLVideoElement>(null);
	const faceDetectorRef = useRef<FaceDetector | null>(null);
	const liveViewRef = useRef<HTMLDivElement>(null);
	const childrenRef = useRef<HTMLElement[]>([]);
	const lastVideoTime = useRef<number>(-1);
	const requestAnimationId = useRef<number | null>(null);
	const faceUndetectedToastId = useRef<string | number | undefined>(undefined);

	const [showInstructions, setShowInstructions] = useState<boolean>(true);
	const [hasReadInstructions, setHasReadInstructions] =
		useState<CheckedState>(false);
	const [startDataCollection, setStartDataCollection] =
		useState<boolean>(false);
	const [gazeDemoCompleted] = useState<boolean>(false);

	const pointDistance = useRef<number>(50);
	const displayPointRadius = useRef<number>(14);

	const startingTimeStampMS = useRef<number>(0);
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
	const webcamMetadataRef = useRef<WebcamMetadata>({
		label: "",
		frame_rate: 0,
		resolution_width: 0,
		resolution_height: 0,
	});

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

					toast.error("No face detected", {
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
						duration: 2000,
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
				for (let xCoord = prevX; xCoord < nextX; xCoord += travelingDistance) {
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
				for (let xCoord = prevX; xCoord > nextX; xCoord -= travelingDistance) {
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
				for (let yCoord = prevY; yCoord < nextY; yCoord += travelingDistance) {
					await drawMovingPoint(
						canvas,
						ctx,
						nextX,
						yCoord,
						movingPointDuration
					);
				}
			} else if (prevY > nextY) {
				for (let yCoord = prevY; yCoord > nextY; yCoord -= travelingDistance) {
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

			// const topLeftPoint = { x: padding, y: padding };
			const topRightPoint = { x: canvasWidth - padding, y: padding };
			const rightMidPoint = {
				x: canvasWidth - padding,
				y: Math.floor(canvasHeight / 2),
			};
			// const bottomLeftPoint = { x: padding, y: canvasHeight - padding };
			// const bottomRightPoint = {
			// 	x: canvasWidth - padding,
			// 	y: canvasHeight - padding,
			// };
			const centerPoint = {
				x: Math.floor(canvasWidth / 2),
				y: Math.floor(canvasHeight / 2),
			};

			const pointOrder = [
				// centerPoint,
				topRightPoint,
				rightMidPoint,
				centerPoint,
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
					baseTravelingPointSpeed,
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
					speed: "normal",
					interval_start_click: false,
					interval_end_click: false,
					interval_start_click_timestamp: 0,
					interval_end_click_timestamp: 0,
				});

				collectedGazeDataRef.current.push({
					x,
					y,
					timestamp: displayedTimestamp,
					event: "stationary",
					speed: "normal",
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

				toast.info("Click the red point.", { duration: requiredClickDuration });

				await new Promise<void>((resolve) => {
					setTimeout(() => {
						resolve();
					}, requiredClickDuration);
				});

				if (
					!usedGroundTruthPointsRef.current[
						usedGroundTruthPointsRef.current.length - 1
					].interval_start_click
				) {
					toast.error("You did not click the point.", {
						duration: requiredClickDuration,
					});
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

				toast.info("Click the red point.", { duration: requiredClickDuration });

				await new Promise<void>((resolve) => {
					setTimeout(() => {
						resolve();
					}, requiredClickDuration);
				});

				if (
					!usedGroundTruthPointsRef.current[
						usedGroundTruthPointsRef.current.length - 1
					].interval_end_click
				) {
					toast.error("You did not click the point.", {
						duration: requiredClickDuration,
					});
				}
			}
		}

		await fullScreenScan();

		if (webcamRef.current && webcamRef.current.srcObject) {
			const stream = webcamRef.current.srcObject as MediaStream;
			const tracks = stream.getTracks();
			tracks.forEach((track) => track.stop());
		}

		setCurrentPhase((prev) => prev + 1);
		return;
	}, [
		pointAccelerationInitiationTime,
		dataCollectionDuration,
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
					setStartDemo(false);
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
				setStartDemo(false);
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

				setStartDemo(false);
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
	}, [setStartDemo]);

	useEffect(() => {
		async function run() {
			initializeGazePointsForCanvas();
			toast.info(
				"The demo has started. Please focus your eyes on the green point.",
				{ duration: 4000 }
			);
			await sleep(6000);
			gazeCollectionLoop();
		}

		if (startDataCollection && !gazeDemoCompleted) {
			run();
		}
	}, [
		startDataCollection,
		gazeDemoCompleted,
		gazeCollectionLoop,
		initializeGazePointsForCanvas,
	]);

	function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
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
							title="Demo Instructions"
							open={showInstructions}
							setOpen={setShowInstructions}
							hasReadInstructions={hasReadInstructions}
							setHasReadInstructions={setHasReadInstructions}
						>
							<div className="text-gray-800">
								<p className="mb-2">
									This is a brief interactive demonstration of how the actual
									study will work. Please read the instructions carefully before
									starting the demo.
								</p>

								<ul className="list-disc pl-5 mb-4">
									<li>
										Your <strong>webcam video feed</strong> is visible in the
										top left corner of the screen as it will be recorded.
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
										During the actual study, if you fail to click all the points
										when they turn red, the study will restart from the
										beginning.
									</li>
									<li className="text-red-600">
										During the actual study, if your face is not visible and
										detected in the webcam feed, the study will restart from the
										beginning.
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
									Please ensure your webcam feed is visible in the top left
									corner before starting the demo.
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
									setStartDataCollection(true);
								}}
								className="flex items-center text-base px-4 py-5"
							>
								<CirclePlayIcon className="h-6 w-6 mr-2" />
								Start Demo
							</Button>
						</div>
					</>
				)
			)}
		</>
	);
}
