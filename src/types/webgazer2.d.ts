declare module "@brandvc/webgazer2" {
	// Add type declarations for the module's exports here
	export function setRegression(
		regressionType: "ridge" | "ridgeWeighted" | "ridgeThreaded"
	): void;
	export function showVideoPreview(show: boolean): void;
	export function showPredictionPoints(show: boolean): void;
	export function applyKalmanFilter(apply: boolean): void;
	export function begin(onFail?: () => void): Promise<void>;
	export function pause(): void;
	export function resume(): Promise<void>;
	export function end(): void;
	export function getStoredPoints(): [number[], number[]];
	export function toggleShowStoredPoints(showPoints: boolean): void;
	export function getStoredPointsWithTimestamp(): [
		number[],
		number[],
		string[]
	];
	export function startStoringPoints(): void;
	export function stopStoringPoints(): void;
	export function setGazeDotColor(color: string): void;
	export function setStoredDotColor(color: string): void;
	export function showFaceOverlay(bool: boolean): void;
	export function showVideo(bool: boolean): void;
	export function showFaceOverlay(bool: boolean): void;
	export function showVideoPreview(val: boolean): void;
	export function showFaceFeedbackBox(bool: boolean): void;

	// Define the type for the params object
	interface CamConstraints {
		video: {
			width: { min: number; ideal: number; max: number };
			height: { min: number; ideal: number; max: number };
			facingMode: string;
		};
	}

	interface Params {
		moveTickSize: number;
		videoContainerId: string;
		videoElementId: string;
		videoElementCanvasId: string;
		faceOverlayId: string;
		faceFeedbackBoxId: string;
		gazeDotId: string;
		videoViewerWidth: number;
		videoViewerHeight: number;
		faceFeedbackBoxRatio: number;
		showVideo: boolean;
		mirrorVideo: boolean;
		showFaceOverlay: boolean;
		showFaceFeedbackBox: boolean;
		showGazeDot: boolean;
		camConstraints: CamConstraints;
		dataTimestep: number;
		showVideoPreview: boolean;
		applyKalmanFilter: boolean;
		saveDataAcrossSessions: boolean;
		storingPoints: boolean;
		showStoredPoints: boolean;
		trackEye: string;
		gazeDotColor: string;
		gazeDotWidth: string;
		gazeDotHeight: string;
		gazeDotOpacity: string;
		gazeDotZIndex: string;
		gazeDotBorderRadius: string;
		storedDotColor: string;
	}

	// Export the params object
	export const params: Params;
}
