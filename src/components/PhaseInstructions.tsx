import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { CheckedState } from "@/types/study";
import React from "react";
import { Checkbox } from "./ui/checkbox";

type PhaseInstructionsProps = {
	children: React.ReactNode;
	title: string;
	open: boolean;
	setOpen: React.Dispatch<React.SetStateAction<boolean>>;
	hasReadInstructions: CheckedState;
	setHasReadInstructions: React.Dispatch<React.SetStateAction<CheckedState>>;
};

export default function PhaseInstructions({
	children,
	title,
	open,
	setOpen,
	hasReadInstructions,
	setHasReadInstructions,
}: PhaseInstructionsProps) {
	return (
		<AlertDialog open={open} onOpenChange={setOpen}>
			<AlertDialogContent className="sm:max-w-xl md:max-w-2xl max-h-[30rem] overflow-y-auto">
				<AlertDialogHeader>
					<AlertDialogTitle className="text-2xl">{title}</AlertDialogTitle>
					<AlertDialogDescription></AlertDialogDescription>
					{children}

					<div className="flex items-center space-x-2 mb-2">
						<Checkbox
							id="terms"
							checked={hasReadInstructions}
							onCheckedChange={setHasReadInstructions}
							required
						/>
						<label
							htmlFor="terms"
							className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
						>
							I have read and understood the instructions
						</label>
					</div>
				</AlertDialogHeader>
				<AlertDialogFooter>
					<AlertDialogAction disabled={!hasReadInstructions}>
						Continue
					</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	);
}
