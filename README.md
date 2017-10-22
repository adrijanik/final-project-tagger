# Morphosyntactic Disambiguer

This is a final project for Data Science Bootcamp 2017 and also a submission for Poleval competition Task 1.

### Task description:
Task 1: POS Tagging
Introduction

There is an ongoing discussion whether the problem of part of speech tagging is already solved, at least for English (see Manning 2011), by reaching the tagging error rates similar or lower than the human inter-annotator agreement, which is ca. 97%. In the case of languages with rich morphology, such as Polish, there is however no doubt that the accuracies of around 91% delivered by taggers leave much to be desired and more work is needed to proclaim this task as solved.

The aim of this proposed task is therefore to stimulate research in potentially new approaches to the problem of POS tagging of Polish, which will allow to close the gap between the tagging accuracy of systems available for English and languages with rich morphology.
Task definition
Subtask (A): Morphosyntactic disambiguation and guessing

Given a sequence of segments, each with a set of possible morphosyntactic interpretations, the goal of the task is to select the correct interpretation for each of the segments and provide an interpretation for segments for which only 'ign' interpretation has been given (segments unknown to the morphosyntactic dictionary).

How to run it?

python main.py
