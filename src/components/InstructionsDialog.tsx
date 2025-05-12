import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogClose,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { CirclePlayIcon } from "lucide-react";
import { useState } from "react";
import { CheckedState } from "@/types/study";
import { toast } from "sonner";

type InstructionsDialogProps = {
	dialogTriggerText: string;
	setStartDemo: React.Dispatch<React.SetStateAction<boolean>>;
	setCurrentPhase: React.Dispatch<React.SetStateAction<number>>;
	children: React.ReactNode;
};

export default function InstructionsDialog({
	children,
	dialogTriggerText,
	setCurrentPhase,
	setStartDemo,
}: InstructionsDialogProps) {
	const [readInstructions, setReadInstructions] = useState<CheckedState>(false);

	return (
		<Dialog
			onOpenChange={(open) => {
				if (!open) {
					setReadInstructions(false);
				}
			}}
		>
			<DialogTrigger asChild>
				<Button className="flex items-center text-base px-4 py-5">
					<CirclePlayIcon className="h-5 w-5 mr-2" />
					{dialogTriggerText}
				</Button>
			</DialogTrigger>
			<DialogContent className="sm:max-w-xl md:max-w-2xl max-h-[30rem] overflow-y-auto">
				<DialogHeader>
					<DialogTitle className="mb-2 text-2xl">Overview</DialogTitle>
					<DialogDescription className="text-gray-800 text-base">
						The study will take approximately 5 to 7 minutes to complete.{" "}
						<span className="font-bold underline">
							Please finish the study in one sitting, avoid refreshing the page,
							resizing the browser window, or navigating away from the study
							once it has begun. Doing so will disrupt the study and you will
							have to start over again.
						</span>
					</DialogDescription>

					<div className="w-full border-b border-gray-900 pt-2"></div>

					{children}

					<div className="flex items-center space-x-2 pt-3 mb-2">
						<Checkbox
							id="instructions"
							checked={readInstructions}
							onCheckedChange={setReadInstructions}
							required
						/>
						<label
							htmlFor="instructions"
							className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
						>
							I have read and understood the instructions
						</label>
					</div>
				</DialogHeader>

				<DialogFooter className="sm:justify-end">
					<DialogClose asChild>
						<Button
							onClick={async () => {
								await navigator.mediaDevices
									.getUserMedia({
										video: true,
										audio: false,
									})
									.then(() => {
										setStartDemo(true);
										setCurrentPhase((prev) => prev + 1);
									})
									.catch(() => {
										toast.error(
											"Unable to access webcam, please make sure that you have a webcam connected and that you have given permission to access it.",
											{
												duration: 6000,
											}
										);
									});
							}}
							type="button"
							disabled={(readInstructions as boolean) === false}
						>
							Begin
						</Button>
					</DialogClose>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
