import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useState } from "react";
// import { supabaseClient } from "@/lib/supabaseClient";
// import { toast } from "sonner";
import { FormInputIcon, GraduationCap, Loader2 } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { useNavigate } from "react-router-dom";
import ConsentInstructions from "@/components/ConsentInstructions";

type ParticipantInfo = {
	firstName: string;
	lastName: string;
	email: string;
	studyCode: string;
};

export default function Validation() {
	const [participantInfo, setParticipantInfo] = useState<ParticipantInfo>({
		firstName: "",
		lastName: "",
		email: "",
		studyCode: "",
	});
	const [isLoading, setIsLoading] = useState(false);
	const [showConsentForm, setShowConsentForm] = useState<boolean>(true);
	const { setState } = useAuthStore();
	const navigate = useNavigate();

	async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
		e.preventDefault();

		setIsLoading(true);

		// const { data, error } = await supabaseClient.from("codes").select();

		setIsLoading(false);

		// if (error) {
		// 	toast.error("An error occurred. Please contact the test adminstrator.");
		// 	return;
		// }

		// const code = data?.find((row) => row.id === participantInfo.studyCode);

		// if (!code) {
		// 	toast.error("Invalid code to start the study.");
		// 	return;
		// }

		// toast.success("Redirecting to the study page.");
		setState({
			firstName: participantInfo.firstName,
			lastName: participantInfo.lastName,
			email: participantInfo.email,
			isAuthenticated: true,
		});

		navigate("/study");
	}

	return (
		<div className="flex items-center justify-center min-h-screen w-full">
			<ConsentInstructions
				title="Study Consent Form"
				open={showConsentForm}
				setOpen={setShowConsentForm}
			>
				<img
					src="uh-horizontal.png"
					alt="University of Houston Logo"
					className="pb-4"
					width="300"
				/>

				<div className="text-gray-800 text-base mb-2">
					<Button variant="outline" asChild className="mb-4">
						{/* <GraduationCap /> */}
						<a
							href="https://www.ece.uh.edu/faculty/nguyen"
							target="_blank"
							className="mb-4 gap-2 font-bold"
						>
							<GraduationCap />
							Dr. Hien Van Nguyen Electrical and Computer Engineering Faculty
							Page
						</a>
					</Button>

					<p className="mb-2">
						The University of Houston Houston Ubiquitous Learning Algorithms
						Research Group (HULA) supervised by Dr. Hien Van Nguyen is
						conducting a study to collect eye gaze data of humans to train and
						develop foundation eye tracking AI models.
					</p>

					<p className="mb-2">
						This study will involve the collection of eye gaze data by recording
						your face and eye movements while performing an online task. This
						data will be publicly released for research purposes and no personal
						information will be made public.
					</p>

					<p className="mb-4">
						If you want to participate in this study, please download, read and
						sign the consent form below. When you have finished you can continue
						with the study and you will be asked to upload the consent form at
						the end of the study.
					</p>

					<p className="mb-4">
						If you have questions or concerns about this study, please contact
						Akash Awasthi at aawasth3@cougarnet.uh.edu or Brandon Chung at
						bvchung@cougarnet.uh.edu
					</p>

					<p className="mb-4 font-bold text-red-700">
						Please complete this study using a Google Chrome or Microsoft Edge
						web browser if possible.
					</p>

					<Button asChild className="mt-2 flex items-center gap-2 w-56">
						<a
							href="https://docs.google.com/document/d/1B80wBGYmI0dn5tfSgCHWpmJaiW8ces6G/edit?usp=sharing&ouid=101185512026423725875&rtpof=true&sd=true"
							target="_blank"
						>
							<FormInputIcon /> Consent Form
						</a>
					</Button>
				</div>
			</ConsentInstructions>

			<form
				onSubmit={handleSubmit}
				className="w-full max-w-sm flex flex-col gap-4"
			>
				<h1 className="text-2xl font-semibold">
					Enter Participant Information
				</h1>

				<div>
					<Label htmlFor="firstName" className="pb-1">
						First Name
					</Label>
					<Input
						id="firstName"
						name="firstName"
						value={participantInfo.firstName}
						onChange={(e) =>
							setParticipantInfo((prev) => ({
								...prev,
								firstName: e.target.value,
							}))
						}
						className="w-full border-gray-400"
						type="text"
						required
					/>
				</div>
				<div>
					<Label htmlFor="lastName" className="pb-1">
						Last Name
					</Label>
					<Input
						id="lastName"
						name="lastName"
						value={participantInfo.lastName}
						onChange={(e) =>
							setParticipantInfo((prev) => ({
								...prev,
								lastName: e.target.value,
							}))
						}
						className="w-full border-gray-400"
						type="text"
						required
					/>
				</div>
				<div>
					<Label htmlFor="email" className="pb-1">
						Email
					</Label>
					<Input
						id="email"
						name="email"
						value={participantInfo.email}
						onChange={(e) =>
							setParticipantInfo((prev) => ({
								...prev,
								email: e.target.value,
							}))
						}
						className="w-full border-gray-400"
						type="email"
						required
					/>
				</div>
				{/* <div>
					<Label htmlFor="studyCode" className="pb-1">
						Authentication Code
					</Label>
					<Input
						id="studyCode"
						name="studyCode"
						value={participantInfo.studyCode}
						onChange={(e) =>
							setParticipantInfo((prev) => ({
								...prev,
								studyCode: e.target.value,
							}))
						}
						className="w-full border-gray-400"
						type="text"
						required
					/>
				</div> */}

				{isLoading ? (
					<Button disabled className="mt-2">
						<Loader2 className="animate-spin h-5 w-5 mr-2" /> Verifying
					</Button>
				) : (
					<Button className="mt-2">Submit</Button>
				)}
			</form>
		</div>
	);
}
