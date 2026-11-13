# Achieving Real-Time Cyber Defense

Source: https://www.boozallen.com/insights/velocity/achieving-real-time-cyber-defense.html

# Booz Allen: AI Cyber Insights

# Achieving Real-Time Cyber Defense

## Bolstering cyber defenses with AI

By Tony Sharp, Joe Gillespie, Matt Costello, Mike Saxton, and Rita Ismaylov

“It was a swift response and a successful containment,” the chief information security officer (CISO) said. But it wasn’t. The security operations center (SOC) intercepted the first intrusion—a carefully targeted spear-phishing email campaign. One executive forwarded the message to the security team along with its skillfully crafted bait and malicious PDF attachment. She told them that it looked like it came from a trusted contact, but the email address was unfamiliar.

Detonated in the SOC’s sandbox, the malware revealed itself, attempting to beacon back to its command-and-control (C2) infrastructure. Armed with the C2 address, the team quickly located the half dozen other malicious emails whose recipients had opened the attachment. Their compromised devices were isolated and disinfected.

Three days later, the team blocked the second intrusion. The attackers used password spray attacks on a misconfigured cloud services portal. Their automated script tried hundreds of logins in rapid succession with email and password combinations culled from previous data breaches. Alerted by one of their identity security tools, the SOC secured the portal and updated their blocklist with the Internet Protocol (IP) addresses from which the login attempts had originated.

As his nighttime counterpart had done earlier with the spear-phishing attack, the daytime SOC manager prepared a report cataloging the attacker’s tactics, techniques, and procedures (TTPs) and the relevant indicators of compromise. He posted the report anonymously through their industry’s information sharing and analysis center to warn other enterprises in their sector about the attacks. “In both of those incidents,” the CISO correctly insisted afterwards, “the team did everything right.” But they still missed the third intrusion.

The fatal vulnerability turned out to be a web server that was spun up five years earlier to support a failed and swiftly canceled series of corporate marketing events. It hadn’t been updated in years, and scans conducted by the security team didn’t flag it because the server used the never-reported and now-forgotten brand of the marketing event, rather than the company’s name, as its domain.

Buried in that server’s file directory were years-old Secure Shell (SSH) keys that still provided trusted access to the organization’s main cluster of marketing servers. Once there, the hackers were able to swiftly pivot and gain domain access to SharePoint and OneDrive. At that point, as revealed in logs they later recovered, there was a 48-hour pause—probably while the hackers sold their access to a ransomware gang.

The SSH connection from the abandoned web server was an “anomalous” event. The security team’s incident and event management platform flagged it as such using the code “amber/anomalous,” the lowest level of alert. The platform flagged hundreds of other amber events that day and dozens classed as “red/suspicious.” None were rated “flashing red/ hostile.” The security team’s half-dozen analysts attempted to resolve as many as they could, but no one checked on the anomalous SSH connection.

“When you have that many false positives, stuff is occasionally going to slip through the cracks,” the CISO acknowledged. “The bottom line is we don’t have the manpower to resolve every anomaly report.”

The postmortem by a third-party security company discovered that the scans of and attacks on the abandoned marketing server originated from the same IP address range the SOC placed on their blocklist because they were the origin of the password spray attacks. There were also links through Whois Database Search to the spear-phishing email campaign. Putting the IP addresses on the blocklist didn’t protect the abandoned server because—as an unrecorded, abandoned shadow IT asset—it was not subject to the company’s security policy.

“If only we’d had a way to sort through those alerts, separating the wheat from the chaff, and the capability to correlate data from the significant ones, we might have spotted the connection and stopped the third attack,” the CISO said in the postmortem internal inquiry. When they returned to the main marketing cluster with domain access after 48 hours, the hackers began exfiltrating and encrypting data.

### Putting Security Teams on the Front Foot

This vignette is fictional, but security personnel routinely deal with attacks like those described above. Today, even well-resourced enterprise cybersecurity teams face an increasingly untenable operational tempo, with little opportunity to do meaningful triage and isolate the most dangerous intrusions. More data about system conditions and potential attacker activity is only useful up to the point where it can be analyzed; beyond this it only increases the size of the haystack concealing the needle of potential threats.

Combine that data overload with the growing complexity of modern enterprise IT systems—multicloud infrastructure, multivendor toolsets, multijurisdictional compliance—and you have a recipe for failure for even the most experienced and best resourced SOC. That recipe is worsening now that attackers can use generative AI (GenAI) to scale personalized, carefully targeted spear-phishing attacks from a handful of targets to hundreds.

The good news is that enterprises can flip the script on current cybersecurity practices. By leveraging AI and machine learning (ML) early in the alert lifecycle, they can quiet the noise of nonstop false positives, optimize their incident response workflow, and help security team members perform critical tasks faster and more efficiently. AI-enhanced intelligence can improve security teams’ ability to conduct proactive threat hunting by using at-scale surveillance of gray space to analyze pre-attack activity and figure out how attackers plan to breach a network.

“AI-enhanced intelligence can improve security teams’ ability to conduct proactive threat hunting by using at-scale surveillance of gray space to analyze pre-attack activity and figure out how attackers plan to breach a network.”

### Overwhelmed on Two Fronts

The irony is that the security team in our fictional example had all the data they needed to discover and halt the security intrusion. They just didn’t know they had it or how to leverage it. This is true in a surprisingly sizable proportion of real-world attacks, and it happens because security teams are increasingly overwhelmed across two fronts: technology density and human resourcing.

These challenges are getting more severe as enterprises confront the growing threats of AI-powered cyberattacks. GenAI bots specifically designed for cybercrime, like Evil GPT or WormGPT, have been advertised on dark web hacker forums for at least two years, and a team of researchers from Indiana University found real-world instances of cybercrime actors using GenAI tools to write malware, generate phishing emails, and set up scam websites.

“AI’s greatest potential use case for cybersecurity may be its power to evolve threat hunting from a large-scale guessing game to an intelligence-driven approach.”

### A Data Problem, Not a Defense Problem

The brutal truth is security teams are often too overwhelmed by their data problem to mount an effective defense. There is too much data arriving too quickly for human team members to timely distinguish relevant threats from irrelevant noise. In addition, security teams don’t have enough insight into how seriously threat actors might seek to compromise their network. By exploiting the growing capabilities of AI for data analysis to focus on relevant threats and draw conclusions about attacker TTPs from at-scale surveillance of pre-attack threat group activity, cybersecurity teams can finally get ahead of their assailants.

### Quieting the Noise

When deployed early in the alert cycle, ML and AI help threat detection teams to better manage the thousands of duplicative and mostly false-positive alerts received daily from various cyber tools.

It’s also possible to use an ML model to prioritize critical alerts and flag them as essential so that analysts don’t need to judge the likelihood of an alert’s importance. Powerful AI algorithms allow organizations to automate this process with confidence, which helps to deprioritize unlikely events and does not rely on humans alone to assess the viability of an alert being fatal.

Integrating these tools into existing workflows and customizing them to meet the needs of the system keeps analysts focused on dealing with true positive alerts, which mitigates risk by decreasing the number of alerts that require human attention and lessening alert fatigue.

Private companies and federal agencies are already using AI to bolster their cyber defenses.

A global automaker, for example, set out to identify gaps in its security posture, formulate a roadmap to get to a more effective security operation, and increase visibility into the overall threat landscape. Scale was a big issue: 1,200 unique data sources of cyber information; numerous sites; and billions of messages from cyber tools every day. Analysts were left to deduce meaningful patterns from unwieldy datasets and spent excessive working hours analyzing false positives.

The automaker addressed these challenges by adding ML and AI early in the alert lifecycle. It also created an easily deployable automated template that could be deployed quickly in any network environment and cloud-based data pipelines large enough to manage the huge volumes of cyber information. The project also set up a cyber-AI capability that provided users across the security team with access to custom big data, ML, and AI intelligence products.

These solutions made an immediate and considerable impact. Instead of dealing with an unwieldy alert feed, the team is able to direct its attention to true positive alerts and guarantee the delivery of security events within seconds from anywhere on the planet.

In the federal government, a large and critical mission needed a more effective way to discover and remediate vulnerabilities. They began piloting a solution that uses AI to pull real-time assessments of system risk into a detection suit. This AI-powered risk analysis helps quantify how an adversary might exploit a given system, which improves the security team’s understanding of risk as part of the monitoring approach. The pilot application has enabled the team to detect cyberattacks that circumvented traditional defenses within seconds while significantly reducing false positives and alert fatigue.

### Breaking Incident Response Bottlenecks

AI and ML can help widen bottlenecks and free up blockages in incident response. For example, certain incidents require the same type of resolution each time they recur. In addition, it’s not uncommon for a familiar type of false positive to hit a network under predictable circumstances. Rather than send these incidents to an analyst’s queue for remediation, large language models (LLM) can identify them and expedite their resolution.

As part of an automated enrichment workflow, an LLM model can be integrated into a security orchestration, automation, and response solution to initiate a chat conversation between the user and the AI. The AI is provided the context of the alert with instructions to gather more information from the user and provide a summary of the conversation, which is added to the case. Separately, an ML algorithm can be layered into this approach to classify the type of incident received as a benign alert or malicious threat based on similarities to previous incidents.

LLMs can also be useful to members of security teams. When added to a tool like Slack, an LLM can update an analyst about the latest ticket remediations or provide a summary of what happened in the last 10 days. From a user experience standpoint, LLMs can have the capacity to reach out to users that provided information in a ticket, a process that currently consumes a significant amount of time for analysts. Rather than dedicate human hours to this process, AI can interact with the user to identify gaps and gather additional information the analyst needs to complete the investigation.

### Pulling Insights from Gray Space

AI’s greatest potential use case for cybersecurity may be its power to evolve threat hunting from a large-scale guessing game to an intelligence-driven approach in which security teams exploit new sources of actionable intelligence to identify and block attacker TTPs before they’re employed against the enterprise.

Before hackers strike, they prepare infrastructure (like fake login sites for phishing campaigns), write or fine-tune malware, and otherwise engage in activity that provides strong indications of planned future attacks. Typically, this activity is not conducted in the attackers’ own infrastructure (“red space”), nor is it deployed at this stage in the defenders’ infrastructure (“blue space”). That comes later, during the attack itself.

Instead, such attack preparation activity commonly takes place in compromised or rented infrastructure temporarily controlled but not owned by the hackers (gray space). Gray space locations can be monitored via the public internet, and many experienced or skilled threat analysts probably track a handful of such locations used for this purpose by threat actors, scouring data from their activities to make informed predictions about future attack campaigns.

Using the growing capabilities of AI, monitoring and analysis work can now be scaled. Internet-wide scans and other comprehensive data-gathering techniques (e.g., analysis of all internet domains established in the last 24 hours) can be used to find evidence of malicious activities, which can then be monitored. AI can draw conclusions from that monitoring data to define TTPs highly likely to be used in future attacks.

The data can then be cross-referenced with information about the enterprise network—its vulnerabilities and system posture—to create detection and prevention methods on the fly. Such methods are derived from using AI/ML to compare the current system posture with the vulnerabilities attackers plan to exploit and devising countermeasures that can be implemented by security tools.

### Empowering Security Teams for Success

Returning to our fictional vignette, as noted, the team had access to all the data they needed to prevent the third attack, but they didn’t know they had it, and they lacked the tools they needed to use that data to defend themselves. Using the approaches we’ve outlined here, our fictional security team would have been set up for success, not failure.

There’s a lot of hype about AI, but these are use cases where judicious application of AI tools can transform the security landscape for large enterprises.

### Key Takeaways

### Subscribe

Want more insights from Velocity ? Sign up to receive more stories about emerging technologies and the impacts they’re making on missions of national importance.

Your preferences enable us to deliver material that is specifically suited to your needs—and you can update your communications choices at any time.

If you are having trouble changing your preferences, please contact us at reply@bah.com .