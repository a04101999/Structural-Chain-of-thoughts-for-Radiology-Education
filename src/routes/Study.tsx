import { useEffect, useState } from "react";
import InstructionsDialog from "@/components/InstructionsDialog";
import {
	// CheckIcon,
	// ClipboardIcon,
	FileDownIcon,
	FormInputIcon,
	Loader2,
	SaveIcon,
	StepForwardIcon,
	WebcamIcon,
} from "lucide-react";
// import VigilanceTest from "@/components/VigilanceTest";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { ParticipantStudyData } from "@/types/study";
import { v4 as uuidv4 } from "uuid";
import TestWebcam from "@/components/TestWebcam";
import { useAuthStore } from "@/store/authStore";
import { useNavigate } from "react-router-dom";
import Gaze from "@/components/Gaze";
import Demo from "@/components/Demo";

async function getDeviceName() {
	if ("getScreenDetails" in window) {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const screenDetails = await (window as any).getScreenDetails();

		return screenDetails.currentScreen.label;
	}

	return "";
}

export default function Study() {
	const { credentials } = useAuthStore();
	const navigate = useNavigate();
	const [isLoading] = useState<boolean>(false);
	const [startSimulation, setStartSimulation] = useState<boolean>(false);
	const [startDemo, setStartDemo] = useState<boolean>(false);
	const [currentPhase, setCurrentPhase] = useState<number>(0);
	const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
	const [isTestingWebcam, setIsTestingWebcam] = useState<boolean>(false);
	const [participantStudyData, setParticipantStudyData] =
		useState<ParticipantStudyData>({
			participant_first_name: credentials.firstName,
			participant_last_name: credentials.lastName,
			participant_email: credentials.email,
			participant_id: `p${uuidv4()}`,
			study_start_date: new Date().toISOString(),
			study_end_date: "",
			device_name: "",
			gaze_study_data: {
				study_duration: 0,
				screen_width: screen.width,
				screen_height: screen.height,
				window_inner_width: window.innerWidth,
				window_inner_height: window.innerHeight,
				window_outer_width: window.outerWidth,
				window_outer_height: window.outerHeight,
				gaze_data: [],
				used_ground_truth_points: [],
				study_start_timestamp: 0,
				study_end_timestamp: 0,
				webcam_metadata: {
					label: "",
					frame_rate: 0,
					resolution_width: 0,
					resolution_height: 0,
				},
			},
			video_recording_metadata: {
				file_name: "",
				recording_start_timestamp: 0,
				recording_end_timestamp: 0,
			},
			survey_completed: false,
		});

	// const [hasCopied, setHasCopied] = useState(false);

	// useEffect(() => {
	// 	setTimeout(() => {
	// 		setHasCopied(false);
	// 	}, 4000);
	// }, [hasCopied]);

	function downloadRecording() {
		if (recordedChunks.length === 0) {
			toast.error("No recording to download.");
			return;
		}

		const blob = new Blob(recordedChunks, {
			type: "video/webm",
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		document.body.appendChild(a);
		a.style.display = "none";
		a.href = url;
		a.download = `${participantStudyData.participant_id}.webm`;
		a.click();
		window.URL.revokeObjectURL(url);
	}

	async function handleSaveData() {
		const studyEndDate = new Date().toISOString();
		const deviceName = await getDeviceName();

		setParticipantStudyData((prev) => ({
			...prev,
			device_name: deviceName,
			study_end_date: studyEndDate,
			survey_completed: true,
		}));

		setCurrentPhase((prev) => prev + 1);

		// setIsLoading(true);

		// try {
		// 	const res = await fetch(`${import.meta.env.VITE_API_URL}/api/results`, {
		// 		method: "POST",
		// 		headers: {
		// 			"Content-Type": "application/json",
		// 		},
		// 		body: JSON.stringify(pData),
		// 	});

		// 	setIsLoading(false);

		// 	if (res.ok) {
		// 		toast.success("Data saved successfully.");
		// 	} else {
		// 		toast.error(
		// 			"Failed to save data, please contact the study administrator."
		// 		);
		// 	}
		// } catch (error) {
		// 	console.error(error);

		// 	toast.error(
		// 		"Failed to save data, please contact the study administrator."
		// 	);
		// } finally {
		// 	setIsLoading(false);
		// 	setCurrentPhase((prev) => prev + 1);
		// }
	}

	useEffect(() => {
		if (!credentials.isAuthenticated) {
			navigate("/");
		}
	}, [credentials, navigate]);

	if (currentPhase === 4) {
		return (
			<div className="w-full h-screen flex flex-col items-center justify-center">
				<div className="max-w-3xl flex flex-col items-center justify-center">
					<h1 className="text-2xl font-semibold">
						The study is almost complete, please click the button below to save
						your recording and results. This will take a moment to process.
					</h1>
					{isLoading ? (
						<Button
							disabled
							className="mt-4 flex items-center gap-2 w-40 text-lg font-medium"
						>
							<Loader2 className="animate-spin h-5 w-5 mr-2" /> Saving
						</Button>
					) : (
						<Button
							onClick={handleSaveData}
							className="mt-4 flex items-center gap-2 w-40 text-base font-medium"
						>
							<SaveIcon /> Save Results
						</Button>
					)}
				</div>
			</div>
		);
	}

	if (currentPhase === 5) {
		return (
			<div className="w-full h-screen flex flex-col items-center justify-center">
				<div className="max-w-3xl flex flex-col items-center">
					<div className="font-medium text-lg">
						<p className="font-semibold text-xl">
							The survey is now complete. Before you exit the study, please
							follow these steps:
						</p>
						<ol className="list-decimal list-inside ">
							<li>
								<strong>
									Download both the recording and study data to submit to the
									Google Form.
								</strong>
							</li>
							<li>
								<strong>Click the link below</strong> to open the Google Form
								and complete the questionnaire.
							</li>
							<li className="font-bold">
								Once you have submitted the form, you may exit this website.
							</li>
						</ol>
					</div>
					<Button
						onClick={() => {
							const fileName = `${participantStudyData.participant_id}.json`;
							const dataStr =
								"data:text/json;charset=utf-8," +
								encodeURIComponent(JSON.stringify(participantStudyData));
							const downloadAnchorNode = document.createElement("a");
							downloadAnchorNode.setAttribute("href", dataStr);
							downloadAnchorNode.setAttribute("download", fileName);
							document.body.appendChild(downloadAnchorNode);
							downloadAnchorNode.click();
							downloadAnchorNode.remove();
						}}
						className="mt-4 flex items-center gap-2 w-56"
					>
						<FileDownIcon /> Download Study Data
					</Button>

					<Button
						onClick={downloadRecording}
						disabled={recordedChunks.length === 0}
						className="mt-4 flex items-center gap-2 w-56"
					>
						<FileDownIcon /> Download Recording
					</Button>

					<Button asChild className="mt-4 flex items-center gap-2 w-56">
						<a href="https://forms.gle/Bj6fD3Ur99wQPmfz7" target="_blank">
							<FormInputIcon /> Google Form
						</a>
					</Button>

					{/* <div className="mt-4 flex items-center justify-between gap-2 w-56 border border-gray-400 py-2 pl-4 pr-2 rounded-md">
						<p className="font-semibold text-gray-900 w-44 overflow-hidden whitespace-nowrap overflow-ellipsis">
							{participantStudyData.participant_id}
						</p>
						<Button
							variant="ghost"
							size="icon"
							className="justify-self-end h-9 w-9"
							onClick={() => {
								navigator.clipboard.writeText(
									participantStudyData.participant_id
								);
								setHasCopied(true);
							}}
						>
							{hasCopied ? (
								<CheckIcon className="stroke-green-600" />
							) : (
								<ClipboardIcon />
							)}
						</Button>
					</div> */}
				</div>
			</div>
		);
	}

	return (
		<div className="relative flex items-center justify-center h-screen w-full">
			{currentPhase === 0 && (
				<div className="flex flex-col items-center justify-center">
					<h1 className="text-gray-900 text-3xl font-semibold text-center mb-4">
						Please Ensure Your Webcam Is Working Before Starting the Study
					</h1>

					<Button
						onClick={() => {
							setIsTestingWebcam((prev) => !prev);
						}}
						className="flex items-center text-base px-4 py-5 mb-4"
						variant={isTestingWebcam ? "destructive" : "default"}
					>
						<WebcamIcon className="h-5 w-5 mr-2" />
						{isTestingWebcam ? "Stop Webcam Test" : "Test Webcam"}
					</Button>

					<TestWebcam
						isTestingWebcam={isTestingWebcam}
						setIsTestingWebcam={setIsTestingWebcam}
					/>

					<InstructionsDialog
						dialogTriggerText="Study Instructions"
						setStartDemo={setStartDemo}
						setCurrentPhase={setCurrentPhase}
					>
						<ol className="list-decimal list-inside text-base text-muted-foreground pt-2">
							<li className="text-red-600 mb-2 font-bold">
								Please maximize your browser window and use Google Chrome or
								Microsoft Edge if possible.
							</li>
							<li className="text-red-600 mb-2 font-bold">
								Ensure that your webcam is working and that you are in a
								well-lit area.
							</li>
							<li className="text-gray-800 mb-2">
								In the first phase of the study, you will be asked to follow a
								moving point on the screen. This phase will last around 5 to 7
								minutes.
							</li>
							<li className="text-gray-800">
								When the study is completed you will be asked to download both
								the study data and recording, and then complete a short survey.
							</li>
						</ol>
					</InstructionsDialog>
				</div>
			)}

			{startDemo && currentPhase === 1 && (
				<Demo
					setCurrentPhase={setCurrentPhase}
					// dataCollectionDuration={2000}
					// pointAccelerationInitiationTime={1000}
					// fixationDuration={4000}
					dataCollectionDuration={240000}
					pointAccelerationInitiationTime={180000}
					fixationDuration={4000}
					requiredPointClickTimeLimit={2000}
					baseTravelingPointSpeed={150}
					acceleratedTravelingPointSpeed={100}
					travelDistance={10}
					setStartDemo={setStartDemo}
					setStartSimulation={setStartSimulation}
					participantStudyData={participantStudyData}
				/>
			)}

			{currentPhase === 2 && (
				<div className="w-full h-screen flex flex-col items-center justify-center">
					<p className="text-2xl font-medium">
						The demo has been completed. Continue with the study.
					</p>
					<Button
						onClick={() => {
							setStartSimulation(true);
							setCurrentPhase((prev) => prev + 1);
						}}
						className="mt-4 flex items-center gap-2"
					>
						<StepForwardIcon /> Continue
					</Button>
				</div>
			)}

			{startSimulation && currentPhase === 3 && (
				<Gaze
					setCurrentPhase={setCurrentPhase}
					setRecordedChunks={setRecordedChunks}
					participantStudyData={participantStudyData}
					setParticipantStudyData={setParticipantStudyData}
					dataCollectionDuration={2000}
					pointAccelerationInitiationTime={1000}
					// dataCollectionDuration={240000}
					// pointAccelerationInitiationTime={180000}
					fixationDuration={4000}
					requiredPointClickTimeLimit={2000}
					baseTravelingPointSpeed={150}
					acceleratedTravelingPointSpeed={100}
					travelDistance={10}
					setStartSimulation={setStartSimulation}
				/>
			)}
		</div>
	);
}
