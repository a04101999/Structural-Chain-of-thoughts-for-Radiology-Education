import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import React, { useState } from "react";
import { Checkbox } from "./ui/checkbox";
import { CheckedState } from "@/types/study";

type PhaseInstructionsProps = {
	children: React.ReactNode;
	title: string;
	open: boolean;
	setOpen: React.Dispatch<React.SetStateAction<boolean>>;
};

export default function ConsentInstructions({
	children,
	title,
	open,
	setOpen,
}: PhaseInstructionsProps) {
	const [acceptTerms, setAcceptTerms] = useState<CheckedState>(false);

	return (
		<AlertDialog open={open} onOpenChange={setOpen}>
			<AlertDialogContent className="sm:max-w-xl md:max-w-2xl max-h-[30rem] overflow-y-auto">
				<AlertDialogHeader>
					<AlertDialogTitle className="text-2xl">{title}</AlertDialogTitle>
					<AlertDialogDescription></AlertDialogDescription>
					{children}
					<div className="flex items-center space-x-2 pt-3 mb-2">
						<Checkbox
							id="terms"
							checked={acceptTerms}
							onCheckedChange={setAcceptTerms}
							required
						/>
						<label
							htmlFor="terms"
							className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
						>
							I have read and agree to the terms and conditions
						</label>
					</div>
				</AlertDialogHeader>
				<AlertDialogFooter>
					<AlertDialogAction disabled={!acceptTerms}>
						Continue
					</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	);
}
