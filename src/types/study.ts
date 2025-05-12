export type Point = {
	x: number;
	y: number;
};

export type GazeData = {
	x: number;
	y: number;
	timestamp: number;
	event: // | "fixate"
	| "stationary"
		| "move"
		| "require_starting_mouse_click"
		| "require_ending_mouse_click"
		| "start_fixation"
		| "end_fixation"
		| "mouse_click";
	speed: string;
};

export type WebPageInfo = {
	width: number;
	height: number;
};

export type GazeStudyData = {
	study_duration: number;
	screen_width: number;
	screen_height: number;
	window_inner_width: number;
	window_inner_height: number;
	window_outer_width: number;
	window_outer_height: number;
	study_start_timestamp: number;
	study_end_timestamp: number;
	gaze_data: GazeData[];
	used_ground_truth_points: UsedGroundTruthPoints[];
	point_logs?: PointLogs[];
	webcam_metadata: WebcamMetadata;
};

export type WebcamMetadata = {
	label: string;
	frame_rate: number;
	resolution_width: number;
	resolution_height: number;
};

export type VideoMetadata = {
	recording_start_timestamp: number;
	recording_end_timestamp: number;
};

export type PsychomotorVigilanceData = {
	start_timestamp: number;
	end_timestamp: number;
	reaction_time: number;
};

// type VigilanceTestData = {
// 	test_duration: number;
// 	test_start_timestamp: number;
// 	test_end_timestamp: number;
// 	interstimulus_intervals: number[];
// 	total_number_of_attempts: number;
// 	data: PsychomotorVigilanceData[];
// 	// successful_attempts: PsychomotorVigilanceData[];
// 	// errors_of_commission: PsychomotorVigilanceData[]; // false starts
// 	// errors_of_omission: PsychomotorVigilanceData[]; // expected reaction time exceeded
// };

export type ParticipantStudyData = {
	participant_id: string;
	participant_first_name: string;
	participant_last_name: string;
	participant_email: string;
	study_start_date: string;
	study_end_date: string;
	device_name: string;
	gaze_study_data: GazeStudyData;
	video_recording_metadata: {
		file_name: string;
		recording_start_timestamp: number;
		recording_end_timestamp: number;
	};
	// before_gaze_vigilance_test_data: VigilanceTestData;
	// after_gaze_vigilance_test_data: VigilanceTestData;
	// vigilance_test_data: VigilanceTestData;
	survey_completed: boolean;
};

export type CheckedState = boolean | "indeterminate";

export type SurveySettings = {
	distanceFromScreen: number;
	displayPointRadius: number;
	pointDistance: number;
	baseDisplayPointDuration: number;
	speedUpDisplayPointDuration: number;
	showGroundTruthPoints: CheckedState;
	overall_study_duration: number;
	accelerated_point_duration: number;
};

export type CurrentDisplayedPoint = {
	x: number;
	y: number;
	circle: Path2D;
	event: // | "fixate"
	| "stationary"
		| "move"
		| "require_starting_mouse_click"
		| "require_ending_mouse_click"
		| "mouse_click";
};

export type UsedGroundTruthPoints = {
	x: number;
	y: number;
	speed: string;
	interval_start_click: boolean;
	interval_end_click: boolean;
	interval_start_click_timestamp: number;
	interval_end_click_timestamp: number;
};

export type PointLogs = {
	x: number;
	y: number;
	timestamp: number;
	prev_timestamp: number;
	timestamp_diff: number;
	point_display_timestamp?: number;
	point_duration?: number;
	speed: string;
	event: string;
};
